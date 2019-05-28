echo "Preparing for running code ........."
echo ""
echo "Creating the output directory....."
if [ ! -d "./Output/" ]; then
  mkdir -p "./Output"
fi
echo "Creating the movie directory....."
if [ ! -d "./movies/" ]; then
  mkdir -p "./movies"
fi

echo "Creating the plot directories....."
if [ ! -d "./Plots/" ]; then
  mkdir -p "./Plots"
  mkdir -p "./Plots/4c"
  mkdir -p "./Plots/4d"
  mkdir -p "./Plots/4d/xy"
  mkdir -p "./Plots/4d/yz"
  mkdir -p "./Plots/4d/xz"

fi

echo "Downloading data files....."
if [ ! -f "randomnumbers.txt" ]; then
  wget "https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt"
fi
if [ ! -f "GRBs.txt" ]; then
  wget "strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt"
fi

if [ ! -f "colliding.hdf5" ]; then
  wget "https://home.strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5"
fi
echo ""
echo "Finished preperations"



echo "Executing code ........."
echo "Executing assigment-1............(around 1 min 30)"
python3 ./Code/assigment_1.py > ./Output/assigment1_out.txt
echo "Executing assigment-2............(around 30 sec)"
python3 ./Code/assigment_2.py > ./Output/assigment2_out.txt
echo "Executing assigment-3............(around 2 sec)"
python3 ./Code/assigment_3.py > ./Output/assigment3_out.txt
echo "Executing assigment-4............(around 1 min 30)"
python3 ./Code/assigment_4.py > ./Output/assigment4_out.txt
echo "Executing assigment-5............(around 30 sec)"
python3 ./Code/assigment_5.py > ./Output/assigment5_out.txt
echo "Executing assigment-6............(around 10 sec)"
python3 ./Code/assigment_6.py > ./Output/assigment6_out.txt
echo "Executing assigment-7............(around 3 sec)"
python3 ./Code/assigment_7.py > ./Output/assigment7_out.txt
echo ""
echo "Creating the movies...."

ffmpeg -r 30 -f image2 -s 64x64 -i ./Plots/4c/4c=%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_4c.mp4
ffmpeg -r 30 -f image2 -s 64x64 -i ./Plots/4d/xy/4d_xy=%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_4d_xy.mp4
ffmpeg -r 30 -f image2 -s 64x64 -i ./Plots/4d/yz/4d_yz=%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_4d_yz.mp4
ffmpeg -r 30 -f image2 -s 64x64 -i ./Plots/4d/xz/4d_xz=%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_4d_xz.mp4

