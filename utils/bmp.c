#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

#pragma pack(push, 1)
typedef struct
{
        uint16_t type;               // Magic identifier "BM" 2
        uint32_t size;               // File size in bytes 4
        uint32_t lambda;          // Not used 2
        uint32_t offset;             // Offset to image data in bytes from beginning of file 4
        uint32_t dib_header_size;    // DIB Header size in bytes 4
        uint32_t width;              // Width of the image 4
        uint32_t height;             // Height of the image 4
        uint16_t planes;             // Number of color planes 2
        uint16_t bits_per_pixel;     // Bits per pixel 2
        uint32_t compression;        // Compression type 4
        uint32_t image_size;         // Image size in bytes 4
        uint32_t x_pixels_per_meter; // Pixels per meter in x axis 4
        uint32_t y_pixels_per_meter; // Pixels per meter in y axis 4
        uint32_t colors_used;        // Number of colors used 4
        uint32_t colors_important;   // Number of important colors 4
} BMPHeader;
#pragma pack(pop)

void write_bmp(const char *filename, BMPHeader header, uint8_t *data)
{
        FILE *outfile;

        // // BMP header
        // header.type = 0x4D42;                       // "BM"
        // header.size = 14 + 40 + width * height * 3; // File size
        // header.reserved1 = 0;
        // header.reserved2 = 0;
        // header.offset = 54;          // Offset to image data
        // header.dib_header_size = 40; // Info header size
        // header.width = width;
        // header.height = height;
        // header.planes = 1;
        // header.bits_per_pixel = 24; // 24-bit color depth
        // header.compression = 0;
        // header.image_size = width * height * 3; // Image size
        // header.x_pixels_per_meter = 0;
        // header.y_pixels_per_meter = 0;
        // header.colors_used = 0;
        // header.colors_important = 0;

        // Open file
        outfile = fopen(filename, "wb");
        if (outfile == NULL)
                return;

        // Write header
        fwrite(&header, sizeof(BMPHeader), 1, outfile);

        // Write image data
        fwrite(data, sizeof(byte), header.image_size, outfile);

        // Close file
        fclose(outfile);
}

uint8_t * read_bmp(const char *filename, BMPHeader *header)
{
        FILE *infile;

        // Open file
        infile = fopen(filename, "rb");
        if (infile == NULL)
        {
                printf("Error opening file %s\n", filename);
                return;
        }

        fread(header, sizeof(BMPHeader), 1, infile);

        // Check if it's a valid BMP file
        if (header->type != 0x4D42)
        {
                printf("Invalid BMP file.\n");
                fclose(infile);
                return;
        }

        // Move file pointer to the beginning of image data
        fseek(infile, header->offset, SEEK_SET);

        // Read image data
        uint8_t *data = (uint8_t *)malloc(header->image_size);
        fread(data, sizeof(uint8_t), header->image_size, infile);
        printf("data 13: %d\n", data[13]);

        // Close file
        fclose(infile);

        printf("Type: %c%c\n", (char)(header->type & 0xFF), (char)(header->type >> 8));
        printf("Size: %u bytes\n", header->size);
        printf("Lambda: %u\n", header->lambda);
        printf("Offset: %u bytes\n", header->offset);
        printf("DIB Header Size: %u bytes\n", header->dib_header_size);
        printf("Width: %d pixels\n", header->width);
        printf("Height: %d pixels\n", header->height);
        printf("Planes: %u\n", header->planes);
        printf("Bits Per Pixel: %u\n", header->bits_per_pixel);
        printf("Compression: %u\n", header->compression);
        printf("Image Size: %u bytes\n", header->image_size);
        printf("X Pixels Per Meter: %d\n", header->x_pixels_per_meter);
        printf("Y Pixels Per Meter: %d\n", header->y_pixels_per_meter);
        printf("Colors Used: %u\n", header->colors_used);
        printf("Colors Important: %u\n", header->colors_important);

        return data;
}

void write_cbmp(const char *filename, BMPHeader *header, cbyte *data, lwe_instance lwe)
{
        FILE *outfile;

        // Open file
        outfile = fopen(filename, "wb");
        if (outfile == NULL)
                return;
        
        header->lambda = lwe.lambda;

        // Write header
        fwrite(header, sizeof(BMPHeader), 1, outfile);

        fseek(outfile, header->offset, SEEK_SET);

        for (int j = 0; j < header->image_size; j++)
        {
                WriteFileCByte(outfile, data[j], lwe);
        }

        // Close file
        fclose(outfile);
}

cbyte * read_cbmp(const char *filename, BMPHeader *header)
{
        FILE *outfile;


        // Open file
        outfile = fopen(filename, "rb");
        if (outfile == NULL)
                return;

        // Write header
        fread(header, sizeof(BMPHeader), 1, outfile);

        cbyte * data = (cbyte *)malloc(sizeof(cbyte) * header->image_size);
        lwe_instance lwe = GenerateLweInstance(header->lambda);

        fseek(outfile, header->offset, SEEK_SET);
        for (int i = 0; i < header->image_size; i++){
                // printf("im here %d \n", i);
                data[i] = ReadFileCByte(outfile, lwe);
        }

        // Close file
        fclose(outfile);

        return data;
}
