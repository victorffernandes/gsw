Estimator

https://github.com/malb/lattice-estimator/

https://github.com/mattflow/cbmp

builds/cipher_bmp.o 20 ./resources/10x10-00FFDE.bmp ./result/10x10-00FFDE.cbmp --use-key "16807 408305 503001 330794 112514 449224 364248 330238 208707 263245"
builds/cipher_bmp.o 20 ./resources/10x10-branco.bmp ./result/10x10-branco.cbmp --use-key "16807 408305 503001 330794 112514 449224 364248 330238 208707 263245"
builds/process_bmp.o 20  ./result/10x10-00FFDE.cbmp ./result/10x10-branco.cbmp ./result/result_xor.cbmp 
builds/read_cipher_bmp.o 20 ./result/result_xor.cbmp ./result/result_xor.bmp  "16807 408305 503001 330794 112514 449224 364248 330238 208707 263245"


builds/read_cipher_bmp.o 20 ./result/10x10-00FFDE.cbmp ./result/10x10-00FFDE.bmp  "16807 408305 503001 330794 112514 449224 364248 330238 208707 263245"
builds/read_cipher_bmp.o 20 ./result/10x10-branco.cbmp ./result/10x10-branco.bmp  "16807 408305 503001 330794 112514 449224 364248 330238 208707 263245"

builds/cipher_bmp.o 10 ./resources/sample_6.bmp ./result/sample_6.cbmp --use-key "423 753 217 42 898 712 728 510 835 77" && 
builds/cipher_bmp.o 10 ./resources/sample_5.bmp ./result/sample_5.cbmp --use-key "423 753 217 42 898 712 728 510 835 77" &&
builds/cipher_bmp.o 10 ./resources/sample_4.bmp ./result/sample_4.cbmp --use-key "423 753 217 42 898 712 728 510 835 77" && 
builds/cipher_bmp.o 10 ./resources/sample_3.bmp ./result/sample_3.cbmp --use-key "423 753 217 42 898 712 728 510 835 77" &&
builds/cipher_bmp.o 10 ./resources/sample_2.bmp ./result/sample_2.cbmp --use-key "423 753 217 42 898 712 728 510 835 77" && 
builds/cipher_bmp.o 10 ./resources/sample_1.bmp ./result/sample_1.cbmp --use-key "423 753 217 42 898 712 728 510 835 77" 
builds/cipher_bmp.o 10 ./resources/rectangle.bmp ./result/rectangle.cbmp --use-key "423 753 217 42 898 712 728 510 835 77"


builds/process_bmp.o ./result/sample_5.cbmp ./result/sample_4.cbmp ./result/result_xor.cbmp 
builds/read_cipher_bmp.o ./result/result_xor.cbmp ./result/result.bmp  "423 753 217 42 898 712 728 510 835 77"

builds/read_cipher_bmp.o ./result/sample_1.cbmp ./result/sample_1.bmp  "423 753 217 42 898 712 728 510 835 77" &&
builds/read_cipher_bmp.o ./result/sample_2.cbmp ./result/sample_2.bmp  "423 753 217 42 898 712 728 510 835 77" &&
builds/read_cipher_bmp.o ./result/sample_3.cbmp ./result/sample_3.bmp  "423 753 217 42 898 712 728 510 835 77" &&
builds/read_cipher_bmp.o ./result/sample_4.cbmp ./result/sample_4.bmp  "423 753 217 42 898 712 728 510 835 77" && 
builds/read_cipher_bmp.o ./result/sample_5.cbmp ./result/sample_5.bmp  "423 753 217 42 898 712 728 510 835 77" &&
builds/read_cipher_bmp.o ./result/sample_6.cbmp ./result/sample_6.bmp  "423 753 217 42 898 712 728 510 835 77"



---- Performance TESTS


----- ENCRYPT ------

lambda=4 - cpu
valgrind --tool=massif --massif-out-file=massif-out/size_8-cpu.out builds/cipher_bmp.o 4 ./resources/size_8.bmp ./result/size_8.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_8_-cpu.out builds/cipher_bmp.o 4 ./resources/size_8_.bmp ./result/size_8_.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_16-cpu.out builds/cipher_bmp.o 4 ./resources/size_16.bmp ./result/size_16.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_16_-cpu.out builds/cipher_bmp.o 4 ./resources/size_16_.bmp ./result/size_16_.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_24-cpu.out builds/cipher_bmp.o 4 ./resources/size_24.bmp ./result/size_24.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_24_-cpu.out builds/cipher_bmp.o 4 ./resources/size_24_.bmp ./result/size_24_.cbmp --use-key "11 10 5 15" cpu && 
valgrind --tool=massif --massif-out-file=massif-out/size_32-cpu.out builds/cipher_bmp.o 4 ./resources/size_32.bmp ./result/size_32.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_32_-cpu.out builds/cipher_bmp.o 4 ./resources/size_32_.bmp ./result/size_32_.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_40-cpu.out builds/cipher_bmp.o 4 ./resources/size_40.bmp ./result/size_40.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_40_-cpu.out builds/cipher_bmp.o 4 ./resources/size_40_.bmp ./result/size_40_.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_48-cpu.out builds/cipher_bmp.o 4 ./resources/size_48.bmp ./result/size_48.cbmp --use-key "11 10 5 15" cpu && 
valgrind --tool=massif --massif-out-file=massif-out/size_48_-cpu.out builds/cipher_bmp.o 4 ./resources/size_48_.bmp ./result/size_48_.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_56-cpu.out builds/cipher_bmp.o 4 ./resources/size_56.bmp ./result/size_56.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_56_-cpu.out builds/cipher_bmp.o 4 ./resources/size_56_.bmp ./result/size_56_.cbmp --use-key "11 10 5 15" cpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_64-cpu.out builds/cipher_bmp.o 4 ./resources/size_64.bmp ./result/size_64.cbmp --use-key "11 10 5 15" cpu && 
valgrind --tool=massif --massif-out-file=massif-out/size_64_-cpu.out builds/cipher_bmp.o 4 ./resources/size_64_.bmp ./result/size_64_.cbmp --use-key "11 10 5 15"

lambda=4 - gpu
valgrind --tool=massif --massif-out-file=massif-out/size_8-gpu.out builds/cipher_bmp.o 4 ./resources/size_8.bmp ./result/size_8.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_8_-gpu.out builds/cipher_bmp.o 4 ./resources/size_8_.bmp ./result/size_8_.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_16-gpu.out builds/cipher_bmp.o 4 ./resources/size_16.bmp ./result/size_16.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_16_-gpu.out builds/cipher_bmp.o 4 ./resources/size_16_.bmp ./result/size_16_.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_24-gpu.out builds/cipher_bmp.o 4 ./resources/size_24.bmp ./result/size_24.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_24_-gpu.out builds/cipher_bmp.o 4 ./resources/size_24_.bmp ./result/size_24_.cbmp --use-key "11 10 5 15" gpu && 
valgrind --tool=massif --massif-out-file=massif-out/size_32-gpu.out builds/cipher_bmp.o 4 ./resources/size_32.bmp ./result/size_32.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_32_-gpu.out builds/cipher_bmp.o 4 ./resources/size_32_.bmp ./result/size_32_.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_40-gpu.out builds/cipher_bmp.o 4 ./resources/size_40.bmp ./result/size_40.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_40_-gpu.out builds/cipher_bmp.o 4 ./resources/size_40_.bmp ./result/size_40_.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_48-gpu.out builds/cipher_bmp.o 4 ./resources/size_48.bmp ./result/size_48.cbmp --use-key "11 10 5 15" gpu && 
valgrind --tool=massif --massif-out-file=massif-out/size_48_-gpu.out builds/cipher_bmp.o 4 ./resources/size_48_.bmp ./result/size_48_.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_56-gpu.out builds/cipher_bmp.o 4 ./resources/size_56.bmp ./result/size_56.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_56_-gpu.out builds/cipher_bmp.o 4 ./resources/size_56_.bmp ./result/size_56_.cbmp --use-key "11 10 5 15" gpu &&
valgrind --tool=massif --massif-out-file=massif-out/size_64-gpu.out builds/cipher_bmp.o 4 ./resources/size_64.bmp ./result/size_64.cbmp --use-key "11 10 5 15" gpu && 
valgrind --tool=massif --massif-out-file=massif-out/size_64_-gpu.out builds/cipher_bmp.o 4 ./resources/size_64_.bmp ./result/size_64_.cbmp --use-key "11 10 5 15"

----- PROCESS ------

cpu
valgrind --tool=massif --massif-out-file=massif-out/result_8_xor.out builds/process_bmp.o ./result/size_8.cbmp ./result/size_8_.cbmp ./result/result_8_xor.cbmp cpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_16_xor.out builds/process_bmp.o ./result/size_16.cbmp ./result/size_16_.cbmp ./result/result_16_xor.cbmp cpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_24_xor.out builds/process_bmp.o ./result/size_24.cbmp ./result/size_24_.cbmp ./result/result_24_xor.cbmp cpu && 
valgrind --tool=massif --massif-out-file=massif-out/result_32_xor.out builds/process_bmp.o ./result/size_32.cbmp ./result/size_32_.cbmp ./result/result_32_xor.cbmp cpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_40_xor.out builds/process_bmp.o ./result/size_40.cbmp ./result/size_40_.cbmp ./result/result_40_xor.cbmp cpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_48_xor.out builds/process_bmp.o ./result/size_48.cbmp ./result/size_48_.cbmp ./result/result_48_xor.cbmp cpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_56_xor.out builds/process_bmp.o ./result/size_56.cbmp ./result/size_56_.cbmp ./result/result_56_xor.cbmp cpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_64_xor.out builds/process_bmp.o ./result/size_64.cbmp ./result/size_64_.cbmp ./result/result_64_xor.cbmp cpu

gpu
valgrind --tool=massif --massif-out-file=massif-out/result_8_xor.out builds/process_bmp.o ./result/size_8.cbmp ./result/size_8_.cbmp ./result/result_8_xor.cbmp gpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_16_xor.out builds/process_bmp.o ./result/size_16.cbmp ./result/size_16_.cbmp ./result/result_16_xor.cbmp gpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_24_xor.out builds/process_bmp.o ./result/size_24.cbmp ./result/size_24_.cbmp ./result/result_24_xor.cbmp gpu && 
valgrind --tool=massif --massif-out-file=massif-out/result_32_xor.out builds/process_bmp.o ./result/size_32.cbmp ./result/size_32_.cbmp ./result/result_32_xor.cbmp gpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_40_xor.out builds/process_bmp.o ./result/size_40.cbmp ./result/size_40_.cbmp ./result/result_40_xor.cbmp gpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_48_xor.out builds/process_bmp.o ./result/size_48.cbmp ./result/size_48_.cbmp ./result/result_48_xor.cbmp gpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_56_xor.out builds/process_bmp.o ./result/size_56.cbmp ./result/size_56_.cbmp ./result/result_56_xor.cbmp gpu &&
valgrind --tool=massif --massif-out-file=massif-out/result_64_xor.out builds/process_bmp.o ./result/size_64.cbmp ./result/size_64_.cbmp ./result/result_64_xor.cbmp gpu


----- DECRYPT ------

builds/read_cipher_bmp.o ./result/result_8_xor.cbmp ./result-bmp/result_8_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_16_xor.cbmp ./result-bmp/result_16_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_24_xor.cbmp ./result-bmp/result_24_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_32_xor.cbmp ./result-bmp/result_32_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_40_xor.cbmp ./result-bmp/result_40_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_48_xor.cbmp ./result-bmp/result_48_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_56_xor.cbmp ./result-bmp/result_56_xor.bmp "11 10 5 15" &&
builds/read_cipher_bmp.o ./result/result_64_xor.cbmp ./result-bmp/result_64_xor.bmp "11 10 5 15"

