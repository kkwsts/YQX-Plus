@echo off

set CHECKPOINT_PATH=../end-to-end-regression/experiments/best_model.pt 

echo Running evaluations...
python eval.py --checkpoint_path %CHECKPOINT_PATH%    
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/baseline-Dexter
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/baseline-virtuosoNet

python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/bvae_featf_ld64_hd128_nh8_nl6_b1.0_g10.0
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/bvae_featf_ld64_hd128_nh8_nl6_b4.0_g10.0
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/bvae_featf_ld64_hd128_nh8_nl6_b10.0_g10.0
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/bvae_featl_ld64_hd128_nh8_nl6_b4.0_g10.0
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/bvae_feats_ld64_hd128_nh8_nl6_b4.0_g10.0
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/flow_featf_hd128_nh8_nl6
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/flow_featf_hd128_nh8_nl6_fmtoptimal_transport
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/flow_featl_hd128_nh8_nl6
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/flow_featl_hd128_nh8_nl6_fmtoptimal_transport
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/flow_feats_hd128_nh8_nl6
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/flow_feats_hd128_nh8_nl6_fmtoptimal_transport
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/gmm_featl_nc16
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/gmm_feats_nc16
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/xgboost_featf_md7_ne1700_lr0.05
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/xgboost_featf_md9_ne2500_lr0.05
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/xgboost_featl_md7_ne1700_lr0.05
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/xgboost_featl_md9_ne2500_lr0.05
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/xgboost_feats_md7_ne1700_lr0.05
python eval.py --checkpoint_path %CHECKPOINT_PATH% --custom_midi_dir ../YQX_result/New_output/xgboost_feats_md9_ne2500_lr0.05


echo Done!
pause