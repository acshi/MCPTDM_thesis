rsync -azP anode0:~/acshikh/selfdriving/results.cache ./results.cache0
rsync -azP anode1:~/acshikh/selfdriving/results.cache ./results.cache1
cat results.cache0 results.cache1 > results.cache
