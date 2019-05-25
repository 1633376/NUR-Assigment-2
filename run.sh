echo "Preparing for running code ........."
echo ""
echo "Creating the output directory....."
if [ ! -d "./Output/" ]; then
  mkdir -p "./Output"
fi

echo "Creating the plot directories....."
if [ ! -d "./Plots/" ]; then
  mkdir -p "./Plots"
  mkdir -p "./Plots/4c"
  mkdir -p "./Plots/4d"
fi

echo "Creating the data directories....."
if [ ! -d "./Data/" ]; then
  mkdir -p "./Data"
fi

echo "Downloading data files....."
wget "https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt"
echo ""

echo "Finished preperations"



echo "Executing code ........."
echo "Executing assigment-1............"
python3 ./Code/assigment_1.py #> ./Output/assigment1_out.txt
#echo "Executing assigment-3............"
#python3 ./Code/assigment_3.py > ./Output/assigment3_out.txt