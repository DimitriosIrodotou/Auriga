
# config: minimum and maximum snapshot numbers to process
minnum=0
maxnum=127

# step (1): determine descendants
for ((i=$minnum; i<$maxnum; i++))
do
  ./B-BaseTree/B-BaseTree B-BaseTree/param.txt $i
done

# step (2): make tree
./B-HaloTrees/B-HaloTrees B-HaloTrees/param.txt

# other: make matching catalog
#for ((i=$minnum; i<=$maxnum; i++))
#do
#./MatchSnaps/MatchSnaps param.txt $i
#done

