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