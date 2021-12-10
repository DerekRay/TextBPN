cd dataset/TD500
rm submit.zip
if [ -d ./submit ]
then 
    echo "submit already exists"
else
    mkdir submit
fi
cp $1/*.txt submit
cd submit/;zip -r  submit.zip * &> ../log.txt ;mv submit.zip ../; cd ../
rm log.txt
python Evaluation_Protocol/script.py -g=gt.zip -s=submit.zip
