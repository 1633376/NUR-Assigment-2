echo "Preparing for running code ........."
echo ""
echo "Creating the output directory....."
if [ ! -d "./Output/" ]; then
  mkdir -p "./Output"
fi

echo "Creating the plot directory....."
if [ ! -d "./Plots/" ]; then
  mkdir -p "./Plots"
fi

echo ""
echo "Finished preparations."
echo ""

echo "Executing code ........."
echo "Executing assigment-1............"
python3 ./Code/assigment1.py > ./Output/assigment1_out.txt
echo "Executing assigment-3............"
python3 ./Code/assigment_3.py > ./Output/assigment3_out.txt