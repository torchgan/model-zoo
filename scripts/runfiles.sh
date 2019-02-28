# Common Functions

sleep_kill() {
    sleep $1
    pid=$(ps | grep "python" | sed "s/\s\+/ /g" | cut -d" " -f2)
    kill -9 $pid
}

# Generative Multi Adversarial Network

echo "Running GMAN"

python ../gman/gman.py --cpu 1 -b 32 -s 16 -e 1 &

sleep_kill 120
