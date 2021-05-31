//
//  demoReduction.c
//  Código para calcular una integral tomado del curso de colfax
//  Primera versión: secuencial
//
//  Created by José Alberto Incera Diéguez on 03/05/17.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(){
    const int nTrials = 5;
    static int arr[4] = {1,2,3,4};
    int x=1, j;
    #pragma opm parallel foo reduction(*:x)
    for(j=0; j<4;j++) {
        x*=arr[j];
    }
    printf("%d\n", x);
}
