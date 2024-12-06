for file in targz/*.tar.gz; do
    tar -xzvf "$file" -C mtx
done