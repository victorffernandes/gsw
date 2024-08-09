Estimator

https://github.com/malb/lattice-estimator/

https://github.com/mattflow/cbmp


Generate Encrypted Files .cbmp

´´
builds/cipher_bmp.o 4 ./resources/noise.bmp ./result/noise.cbmp --use-key "11 10 5 15" cpu &&
builds/cipher_bmp.o 4 ./resources/sonic.bmp ./result/sonic.cbmp --use-key "11 10 5 15" cpu
´´

Process Images

builds/process_bmp.o ./result/noise.cbmp ./result/sonic.cbmp ./result/noisy_sonic.cbmp cpu && 
builds/process_bmp.o ./result/noise.cbmp ./result/noisy_sonic.cbmp ./result/sonic.cbmp cpu

Decrypt Images

builds/read_cipher_bmp.o ./result/noisy_sonic.cbmp ./result-bmp/noisy_sonic.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/noise.cbmp ./result-bmp/noise.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/sonic.cbmp ./result-bmp/sonic.bmp "11 10 5 15"

