# Fetches all corpora from their source and puts them in the dat directory

echo "Creating directories"
mkdir -p dat
mkdir -p dat/greek
mkdir -p dat/greek/raw_data

cd dat/greek/raw_data/
echo "Fetching raw corpora"

echo " - Cloning Perseus"
git clone "https://github.com/PerseusDL/canonical-greekLit"
echo " - Cloning First1K Greek"
git clone "https://github.com/OpenGreekAndLatin/First1KGreek"
echo " - Cloning Online Critical Pseudepigrapha"
git clone "https://github.com/OnlineCriticalPseudepigrapha/Online-Critical-Pseudepigrapha.git"

cd ..

echo "Trying to unzip Septuagint data"
sudo apt install unzip
unzip "SEPA.zip" -d "raw_data/SEPA"
