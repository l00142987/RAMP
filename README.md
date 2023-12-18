# Robust Anomalous Sound Detection for Cyclostationary Rotating Equipment Based on Mechanical Sound Extraction and Model Pre-training
The core code released for the paper named "Robust Anomalous Sound Detection for Cyclostationary Rotating Equipment Based on Mechanical Sound Extraction and Model Pre-training“.
### Road map

![](https://github.com/kuper7/ASD-Based-on-MSE-and-Model-Pretraining/blob/main/fig/road_map.png)

------------------------------------------------------------------

### Dataset
Download the MIMII dataset from here : [(zenodo.org)](https://zenodo.org/records/3384388)

Download the AudioSet dataset from here : [(zenodo.org)](https://zenodo.org/records/3384388) ( **The music label in Wikidata is "/m/04rlf"**)
  
Download the NOISEX-92 dataset from here : [(http://spib.linse.ufsc.br/noise.html)]

------------------------------------------------------------------

### Description of the core code

1. Mechanical_Sound_Extraction.py

  In this file, a class *Mechanical_Sound_Extraction* is defined, in which:

* *process_signal(self,X,Q,stop_condition)* is used to process a mechanical sound,

* *X* is the mechanical sound to be processed, and *Q* is the result of the computation using the formula *Q=M×L_max*, which represents the range of MSE's receptive field. 

*  *stop_condition* is the MSE stop condition obtained from the SNR estimation model, calculated by the formula: *SNE_est <= 10log10(E_decomp/E_r)*.
  
* *Eliminate_SNR_errors(self,stop_condition,snr_list,projected_result,projected_save_last,M_)* is used to eliminate SNE estimation errors, where:

* *snr_list* is the sequence of SNR changes obtained by *process_signal* processing;

*  *projected_result* is the initial MSE result obtained by *process_signal* processing;

*  *projected_save_last* is the result of the penultimate Ramanujan subspace projection processing;

*  *M_* is the number of divisions between neighboring SNRs, the larger its value, the higher the division.

2. conformer_AE.py

* In this file, a conformer-based Auto-encoder model is defined based on *conformer_block*.

3. BLSTM_SNR.py

* In this file, a SNR estimation model is defined based on BLSTM.

-------------------------------------------------------------------
***After considering the current progress of the project and some sensitivity issues, we have decided to open source a portion of the core code of our methodology. These codes do not cover all the details of the whole system, but they are sufficient to provide an in-depth understanding of the core principles and implementation of the proposed methodology. All the code will be open-sourced as soon as the paper is accepted.***
