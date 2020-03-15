dir=bike_translation_slow
echo "Testing $dir"
echo "Expect rotation median <2"
echo "Translate median <0.1"
echo ""

python camtrack.py "dataset/$dir/rgb/*" dataset/$dir/camera.yml track.yml point_cloud.yml --load-corners dataset/$dir/corners >/dev/null
python cmptrack.py dataset/$dir/ground_truth.yml track.yml

for dir in "fox_head_short" "fox_head_full" "room"; do
  echo "################################################"
  echo "Testing $dir"
  echo "Expect rotation median <10"
  echo "Rotation max <20"
  echo "Translate median <0.15"
  echo "Translate max <0.3"
  python camtrack.py "dataset/$dir/rgb.mov" dataset/$dir/camera.yml track.yml point_cloud.yml --load-corners dataset/$dir/corners >/dev/null
  python cmptrack.py dataset/$dir/ground_truth.yml track.yml
done


for dir in "ironman_translation_fast" "house_free_motion" "soda_free_motion"; do
  echo "################################################"
  echo "Testing $dir"
  echo "Expect rotation median <10"
  echo "Rotation max <20"
  echo "Translate median <0.15"
  echo "Translate max <0.3"

  python camtrack.py "dataset/$dir/rgb/*" dataset/$dir/camera.yml track.yml point_cloud.yml --load-corners dataset/$dir/corners >/dev/null
  python cmptrack.py dataset/$dir/ground_truth.yml track.yml
done
