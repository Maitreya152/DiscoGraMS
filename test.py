import torch
import torch.nn as nn
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from train import Architechture, generate_tgt_mask
from transformers import LongformerTokenizer, LongformerModel
from train import LazyDataLoader, retreive_graph_script_summary, extract_relevant_content, NUM_EPOCHS, MODEL_DIM, LONGFORMER_MODEL, MINILM_ENC_DIM, MODEL_SAVE_PATH, VOCAB_SIZE

SCRIPTS_PATH = ""
GRAPHS_PATH = ""

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model = Architechture(MINILM_ENC_DIM, MODEL_DIM, MODEL_DIM, VOCAB_SIZE).to(device)
for epoch in range(NUM_EPOCHS):
    test_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}_{epoch+1}.pt"))
    test_model.eval()
    print(f"Model at epoch {epoch + 1} loaded successfully")
    longformer_tokenizer = LongformerTokenizer.from_pretrained(LONGFORMER_MODEL)
    data_loader = LazyDataLoader(SCRIPTS_PATH, GRAPHS_PATH)
    longformer_model = LongformerModel.from_pretrained(LONGFORMER_MODEL).to(device)
    longformer_model.config.attention_mode = 'sliding_chunks'
    longformer_tokenizer.model_max_length = 4096
    print("Testing the model")
    rouge1_total, rouge2_total, rougel_total = 0, 0, 0
    bert_f1_total, bert_p_total, bert_r_total = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    epoch_loss = 0
    for i in tqdm(range(len(data_loader)), desc=f"Testing the model at epoch {epoch+1}/{NUM_EPOCHS}"):
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
            x_out = test_model(pyg_graph, script, summary_encode_input, summary_encode_output, tgt_mask)
        x_out = x_out.permute(1, 0, 2)
        x_out = x_out.view(-1, x_out.size(-1))
        summary_tokens_output = summary_tokens_output.long()
        summary_tokens_output = summary_tokens_output.view(-1)
        loss = loss_fn(x_out, summary_tokens_output)
        epoch_loss += loss.item()

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

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {epoch_loss/len(data_loader)}")
    print(f"ROUGE-1: {rouge1_total/len(data_loader)}")
    print(f"ROUGE-2: {rouge2_total/len(data_loader)}")
    print(f"ROUGE-L: {rougel_total/len(data_loader)}")
    print(f"BERTScore-Precision: {bert_p_total/len(data_loader)}")
    print(f"BERTScore-Recall: {bert_r_total/len(data_loader)}")
    print(f"BERTScore-F1: {bert_f1_total/len(data_loader)}")

print("Testing Completed")
