import json
import torch
import pickle
import wandb
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from rouge_score import rouge_scorer
from torch_geometric.data import Data
from bert_score import score as bert_score
from torch_geometric.nn import GATConv, global_add_pool
from transformers import LongformerModel, LongformerTokenizer

MODEL_SAVE_PATH = ""
NUM_EPOCHS = 20
ENCODER_DIM = 1024
HIDDEN_DIM = 2048
MODEL_DIM = 4096
LONGFORMER_MODEL = 'allenai/longformer-large-4096'
MINILM_ENC_DIM = 768
VOCAB_SIZE = 50265
LEARNING_RATE = 0.00001
WANDB_PROJECT_NAME = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPTS_PATH = ""
GRAPHS_PATH = ""

def generate_tgt_mask(tgt_seq_len):
    return torch.triu(torch.ones((tgt_seq_len, tgt_seq_len)) * float('-inf'), diagonal=1)

class LED_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.longformer_model = LongformerModel.from_pretrained(LONGFORMER_MODEL).to(device)
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained(LONGFORMER_MODEL)
        self.longformer_tokenizer.model_max_length = 4096
        self.longformer_model.config.attention_mode = 'sliding_chunks'
        self.attention = nn.MultiheadAttention(embed_dim=ENCODER_DIM, num_heads=8, dropout=0.1)
        self.attention_pool = nn.Parameter(torch.randn(ENCODER_DIM))
        self.layer_norm = nn.LayerNorm(ENCODER_DIM)
        self.fc1 = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, MODEL_DIM)
        self.relu = nn.ReLU()

    def forward(self, script):
        
        input_ids = self.longformer_tokenizer.encode(script)
        chunks = self.chunk_ids(input_ids, self.longformer_tokenizer.model_max_length)
        max_length = len(chunks)*self.longformer_tokenizer.model_max_length
        input_ids = self.longformer_tokenizer.encode(script, max_length=max_length, padding='max_length')
        chunks = self.chunk_ids(input_ids, self.longformer_tokenizer.model_max_length)
        embeddings = []
        for chunk in chunks:
            input_ids = torch.tensor(chunk).unsqueeze(0)
            input_ids = input_ids.to(device)
            with torch.no_grad():
                output = self.longformer_model(input_ids)
            embeddings.append(output.last_hidden_state)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.view(-1, embeddings.size(-1))
        embeddings = embeddings.unsqueeze(0)
        attention_output, _ = self.attention(embeddings, embeddings, embeddings)
        attn_weights = torch.matmul(attention_output, self.attention_pool)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_sum = torch.matmul(attn_weights, attention_output.squeeze(0))
        output = self.layer_norm(weighted_sum + embeddings.squeeze(0).mean(dim=0))
        weighted_sum = self.fc1(weighted_sum)
        output = self.fc1(output)
        output = self.relu(output) 
        output = self.fc2(output + weighted_sum)
        return output

    def chunk_ids(self, input_ids, chunk_size):
        return [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]

class Graph_Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

    def forward(self, data):
        
        x, edge_index= data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        batch = batch.to(device)
        x = global_add_pool(x, batch)
        return x
    
class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_layer = nn.Linear(2 * MODEL_DIM, MODEL_DIM)
        self.relu = nn.ReLU()
        
    def forward(self, x_led, x_graph):
        
        x = torch.cat((x_led, x_graph), dim=-1)
        x = self.linear_layer(x)
        x = self.relu(x)
        return x
    
class TransformerDecoderModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_layers=6, nhead=8, dim_feedforward=8192, max_len=2284):
        super(TransformerDecoderModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.positional_encoding = nn.Embedding(max_len, ENCODER_DIM)
        self.redim_tgt = nn.Linear(ENCODER_DIM, embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, memory, tgt, tgt_mask, tgt_key_padding_mask=None):
        
        batch_size, tgt_seq_len, _ = tgt.size()
        pos = torch.arange(0, tgt_seq_len).unsqueeze(0).repeat(batch_size, 1).to(tgt.device)
        tgt = tgt + self.positional_encoding(pos)
        tgt = self.redim_tgt(tgt)
        memory = memory.unsqueeze(1)
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc_out(output)
        return self.softmax(output)

class Architechture(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim, vocab_size):
        super().__init__()
        
        self.led_encoder = LED_Encoder().to(device)
        self.gan = Graph_Encoder(in_channels, hidden_channels).to(device)
        self.ffnn = FeedForwardNN().to(device)
        self.transformer_decoder = TransformerDecoderModel(embedding_dim, vocab_size).to(device)

    def forward(self, data, script, tgt_input, tgt_output, tgt_mask):
        
        x_led = self.led_encoder(script)
        x_graph = self.gan(data)
        x_combined = self.ffnn(x_led, x_graph)
        x_out = self.transformer_decoder(x_combined, tgt_input, tgt_mask)
        return x_out

class LazyDataLoader:
    def __init__(self, jsonl_path, pkl_path):
        
        self.jsonl_path = jsonl_path
        self.pkl_path = pkl_path
        self.movie_graphs = None

    def __len__(self):
        
        with open(self.jsonl_path, 'r') as f:
            return sum(1 for _ in f)

    def load_movie_graphs(self):
        
        if self.movie_graphs is None:
            with open(self.pkl_path, "rb") as f:
                self.movie_graphs = pickle.load(f)
        else:
            return
    
    def get_item(self, index):

        script = None
        summary = None
        pyg_graph = None
        with open(self.jsonl_path, "r") as jsonl_file:
            for i, line in enumerate(jsonl_file):
                if i == index:
                    json_obj = json.loads(line)
                    script = json_obj['script']
                    summary = json_obj['summary']
                    break
        self.load_movie_graphs()
        G = nx.Graph()
        G = self.movie_graphs[f"M_{index+1}"]
        node_idx_mapping = {}
        node_types = []
        node_type_mapping = {'scene': 0, 'dialogue': 1, 'character': 2}
        node_features = []
        edges = []
        for i, (node, attr) in enumerate(G.nodes(data=True)):
            node_idx_mapping[node] = i
            node_types.append(node_type_mapping[attr['type']])
            if 'text' in attr:
                node_features.append(torch.tensor(attr['text'], dtype=torch.float32))
            else:
                node_features.append(torch.zeros((768,)))
        x = torch.stack(node_features)
        for edge in G.edges:
            edges.append([node_idx_mapping[edge[0]], node_idx_mapping[edge[1]]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        pyg_graph = Data(x=x, edge_index=edge_index)
        return pyg_graph, script, summary

def retreive_graph_script_summary(index, data_loader):
    return data_loader.get_item(index)

def extract_relevant_content(xml_string):

    root = ET.fromstring(xml_string)
    relevant_text = []
    for scene in root.findall('scene'):
        for elem in scene:
            if elem.tag in ['stage_direction','scene_description', 'character', 'dialogue'] and elem.text is not None:
                relevant_text.append(elem.text.strip()) 
    return '\n'.join(relevant_text)

if __name__ == "__main__":

    wandb.init(project=WANDB_PROJECT_NAME)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    longformer_tokenizer = LongformerTokenizer.from_pretrained(LONGFORMER_MODEL)
    data_loader = LazyDataLoader(SCRIPTS_PATH, GRAPHS_PATH)
    longformer_model = LongformerModel.from_pretrained(LONGFORMER_MODEL).to(device)
    longformer_model.config.attention_mode = 'sliding_chunks'
    longformer_tokenizer.model_max_length = 4096
    loss_fn = nn.CrossEntropyLoss()
    model_architecture = Architechture(MINILM_ENC_DIM, MODEL_DIM, MODEL_DIM, VOCAB_SIZE).to(device)
    optim = optim.Adam(model_architecture.parameters(), lr=LEARNING_RATE)

    print("Training the model")
    model_architecture.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        rouge1_total, rouge2_total, rougel_total = 0, 0, 0
        bert_f1_total, bert_p_total, bert_r_total = 0, 0, 0

        for i in tqdm(range(len(data_loader)), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            optim.zero_grad()
            pyg_graph, script, summary = retreive_graph_script_summary(i, data_loader)
            script = extract_relevant_content(script)
            pyg_graph = pyg_graph.to(device)
            summary_tokens = torch.tensor(longformer_tokenizer.encode(summary)).to(device).unsqueeze(0)
            with torch.no_grad():
                summary_encode = longformer_model(summary_tokens)
            summary_encode = summary_encode.last_hidden_state
            summary_encode_input = summary_encode[:, :-1, :]
            summary_encode_output = summary_encode[:, 1:, :]
            summary_tokens_output = summary_tokens[:, 1:]
            tgt_mask = generate_tgt_mask(summary_encode_input.size(1)).to(device)
            x_out = model_architecture(pyg_graph, script, summary_encode_input, summary_encode_output, tgt_mask)
            x_out = x_out.permute(1, 0, 2)
            x_out = x_out.view(-1, x_out.size(-1))
            summary_tokens_output = summary_tokens_output.long()
            summary_tokens_output = summary_tokens_output.view(-1)
            loss = loss_fn(x_out, summary_tokens_output)
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            predicted_summary_tokens = torch.argmax(x_out, dim=-1)
            predicted_summary = longformer_tokenizer.decode(predicted_summary_tokens.cpu().numpy(), skip_special_tokens=True)
            rouge_scores = scorer.score(summary, predicted_summary)
            rouge1_total += rouge_scores['rouge1'].fmeasure
            rouge2_total += rouge_scores['rouge2'].fmeasure
            rougel_total += rouge_scores['rougeL'].fmeasure
            P, R, F1 = bert_score([predicted_summary], [summary], lang="en", device=device)
            bert_p_total += P.mean().item()
            bert_r_total += R.mean().item()
            bert_f1_total += F1.mean().item()

        wandb.log({
            "epoch": epoch+1,
            "loss": epoch_loss/len(data_loader),
            "ROUGE1": rouge1_total/len(data_loader),
            "ROUGE2": rouge2_total/len(data_loader),
            "ROUGEL": rougel_total/len(data_loader),
            "BERTScore Precision": bert_p_total/len(data_loader),
            "BERTScore Recall": bert_r_total/len(data_loader),
            "BERTScore F1": bert_f1_total/len(data_loader)
        })
        torch.save(model_architecture.state_dict(), f"{MODEL_SAVE_PATH}_{epoch+1}.pt")

    print("Training Completed")
    wandb.finish()
