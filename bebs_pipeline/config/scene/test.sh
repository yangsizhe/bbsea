for (( i=0; i<=30; i++ ))
do
   j=$((i+2))
   if [ -e scene$i.yaml ]; then
       mv scene$i.yaml $j.yaml
   else
       echo "File scene$i.yaml does not exist."
   fi
done

# for (( i=0; i<=30; i++ ))
# do
#    j=$((i+2))
#    if [ -e $i.yaml ]; then
#        mv $i.yaml $j.yaml
#    else
#        echo "File $i.yaml does not exist."
#    fi
# done