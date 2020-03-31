# for file in ../../Data/*/*/*_protein.pdb; do
#     echo "obabel -ipdb $file -omol2 -O ${file%.*}.mol2"
# done > run.sh

# cat run.sh | parallel -j 56 {}

# rm -rf run.sh




for file in ../../Data/*/*/*_ligand.mol2; do
    echo "obabel -imol2 $file -opdb -O ${file%.*}.pdb"
done > run.sh

cat run.sh | parallel -j 56 {}

rm -rf run.sh
