#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int ELEM_PER_VECTOR = 32;

int main()
{
	int n;

	srand( ( unsigned )&n );

	printf( "Enter a size: " );
	scanf( "%d", &n );

	int size = ELEM_PER_VECTOR * n;
	float* vec = new float[ size ];

	for( int i = 0; i < size; ++i )
	{
		vec[ i ] = ( float( rand() ) * 2.f / RAND_MAX ) - 1.f;
	}

	float (* mat)[ELEM_PER_VECTOR] = new float[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ];
	for( int i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			mat[ i ][ j ] = ( float( rand() ) * 2.f / RAND_MAX ) - 1.f;
		}
	}

	FILE* fp = fopen( "gen.bin", "wb" );
	fwrite( &n, sizeof( float ), 1, fp );
	fwrite( vec, sizeof( float ), size, fp );
	fwrite( mat, sizeof( float ), ELEM_PER_VECTOR * ELEM_PER_VECTOR, fp );
	fclose( fp );

	fp = fopen( "gen.bin", "rb" );
	float* vec2 = new float[ size ];
	float( *mat2 )[ ELEM_PER_VECTOR ] = new float[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ];
	int m;
	fread( &m, sizeof( float ), 1, fp );
	fread( vec2, sizeof( float ), m * ELEM_PER_VECTOR, fp );
	fread( mat2, sizeof( float ), ELEM_PER_VECTOR * ELEM_PER_VECTOR, fp );

	if( n != m ) printf( "error: size diff. %n != %n", n, m );
	for( int i = 0; i < size; ++i )
	{
		if( vec[ i ] != vec2[ i ] )
		{
			printf( "[%d] %f != %f\n", vec[ i ], vec2[ i ] );
			break;
		}
	}
	for( int i = 0; i < ELEM_PER_VECTOR; ++i )
	{
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			if( mat[ i ][ j ] != mat2[ i ][ j ] )
			{
				printf( "[%d][%d]\n", i, j );
				break;
			}
		}
	}
	fclose( fp );

	delete[] vec;

	return 0;
}