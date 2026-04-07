start1=$((28 + 5))
start2=$((28 + 5))
count1=20292
count2=888
count3=8000
count4=9999
# Correct sed command for A.pdb, assuming you want to start from line 6 to start1
sed -n "6,${start1}p" interface.pdb > A.pdb

# Correct sed command for B.pdb, using calculated start2 and count1
sed -n "$((start2 + 1)),$((start2 + count1))p" interface.pdb > B.pdb

sed -n "$((start2 + count1 + 1)),$((start2 + count1+ count2))p" interface.pdb > C.pdb

sed -n "$((start2 + count1+count2 + 1)),$((start2 + count1+ count2+count3))p" interface.pdb > D.pdb


#sed -n "$((start2 + count1+count2+count3 + 1)),$((start2 + count1+ count2+count3+count4))p" interface.pdb > E.pdb




