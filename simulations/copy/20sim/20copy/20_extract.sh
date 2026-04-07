last_atom=$(grep "^ATOM" single_polymer.pdb | tail -1 | awk '{print $2}')

for i in {1..20}
do
    sed -n $(((i-1)*last_atom+6)),$((i*last_atom+5))p 20_interface.pdb > A_$i.pdb
done


sed -n "$((20*last_atom + 6)),$((20*last_atom+5 + 9999))p" 20_interface.pdb > BP41.pdb

sed -n "$((20*last_atom+5+9999 + 1)),$((20*last_atom+5+9999 + 2801))p" 20_interface.pdb > BP42.pdb



for i in {1..13}
do
    sed -n $(((i-1)*9999+20*last_atom+5+9999+2802)),$((i*9999+20*last_atom+5+9999+2801))p 20_interface.pdb > W_$i.pdb
done










