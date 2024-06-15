## Requirements

  Install Python 3.9.0+ and the related dependencies using: 

  ``` bash
  pip install -r requirements.txt
  ```

## Usage

1) Run the `main.py` file giving as input the path of the folder of the audio you want to analyse and extract the features to put them into a csv (choosing the output file name)

2) If you want to directly parse a test csv that I've conducted several experiments with, run `RandomClassifier.py`, it will parse the csv and the audio features giving as output 
  -the accuracy in recognising fake and real audio
  -a histogram (with the most relevant features of the analysed audio) 
  -a shap interaction values graph
  -a confusion matrix 

Example:

```bash
python RandomClassifier.py
Enter the file path (only csv): CSV/csv con LABEL/extracted_features_DEEP.csv
```
## Reference
[Is synthetic voice detection research going into the right direction?](https://paperswithcode.com/paper/is-synthetic-voice-detection-research-going)

if this paper helped you in the research, please cite: 

```BibTex
@inproceedings{borzi2022synthetic,
  title={Is Synthetic Voice Detection Research Going Into the Right Direction?},
  author={Borz{\`\i}, Stefano and Giudice, Oliver and Stanco, Filippo and Allegra, Dario},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={71--80},
  year={2022}
}
```
## Credits
- Lorenzo Mongelli
- Dario Allegra
- Stefano Borzì

2024


