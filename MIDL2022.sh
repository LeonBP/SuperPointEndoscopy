#python train4.py train_joint configs/superpoint_uz_train_heatmap.yaml superpoint_ucluzlabel100 --eval --debug
python train4.py train_joint configs/superpoint_uz_train_heatmap_spec.yaml superpoint_ucluzlabel100_spec100pixels --eval --debug
sh match_eval.sh
