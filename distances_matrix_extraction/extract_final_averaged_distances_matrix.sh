#rm -r ../distance_matrices
#mkdir ../distance_matrices
#sleep 3
python extract_tapered_levenshtein_distances_matrix.py $1
sleep 3
#python extract_euclidean_distances_matrix.py
#python extract_gower_distances_matrix.py
python extract_manhattan_distances_matrix.py $1
sleep 3
python extract_final_averaged_distances_matrix.py $1

rm *.pyc
rm -r logs
