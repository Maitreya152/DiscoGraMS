# DiscoGraMS
DiscoGraMS: Enhancing Movie Screen-Play Summarization using Movie Character-Aware Discourse Graph

## Instructions for Running the Code  

1. **Set Hyperparameters:**  
   - Configure the appropriate hyperparameters in both `train.py` and `test.py` before execution.  

2. **Download and Prepare CaD Graphs:**  
   - Download the provided CaD graphs.  
   - Encode the text of each node using a sentence encoder of choice (or refer to the paper for our choice). This step is essential for `train.py` and `test.py` to function correctly.  

3. **Execute the Scripts:**  
   - Once the above steps are completed, proceed with running `train.py` and `test.py`.

## Link to Paper
https://arxiv.org/abs/2410.14666

## Link to Dataset
https://huggingface.co/datasets/Maitreya152/CaD_Graphs

## Reference

## Reference  

If you make use of our work in your research, please cite the following in your manuscript:  

```bibtex
@misc{chitale2025discogramsenhancingmoviescreenplay,
      title={DiscoGraMS: Enhancing Movie Screen-Play Summarization using Movie Character-Aware Discourse Graph}, 
      author={Maitreya Prafulla Chitale and Uday Bindal and Rajakrishnan Rajkumar and Rahul Mishra},
      year={2025},
      eprint={2410.14666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.14666}, 
}
