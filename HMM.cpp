#include <iostream>

using namespace std;
const int N = 3, M = 3, T = 7; // total State N, Number of Observation M, T state
double PI[N] = {0.5, 0.2, 0.3}; // Initial Probability
// Transition Probability
double a[N][N] 	= {	{0.6, 0.2, 0.2},
					{0.5, 0.3, 0.2},
					{0.4, 0.1, 0.5}};
// Observation Probability
double b[N][M]	= {	{0.7, 0.1, 0.2},
					{0.1, 0.6, 0.3},
					{0.3, 0.3, 0.4}};

double δ[T][N];
int ψ[T][N];					
int Observation [T] = {0, 0, 2, 1, 2, 1, 0};

double HMM();
int main(void){
	HMM();
	return 0;
}

double HMM_forward(int T, int * ObservationState){
	double alpha[T][N];
	for(int t = 0; t < T; t++){
		for(int j = 0; j < N; j++){
			if (t == 0)
				alpha[t][j] = PI[j] * b[j][ObservationState[t]];
			else{
				double p = 0;
				for(int i = 0; i < N; i++)
					p += alpha[t-1][i] * a[i][j];
				alpha[t][j] = p * b[j][ObservationState[t]];
			}
		}
	}
	double p = 0;
	for(int i = 0; i < N; i++)
		p += alpha[T -1][i];
	return p;
}

double HMM_backward(int T, int * ObservationState){
	double beta[T][N];
	for(int t = T-1; t >=0; t--){
        for(int i = 0; i < N; i++){
            if(t == T-1)
                beta[t][i] = 1.0;
            else{
                double  p = 0;
                for(int j = 0; j < N; j++)
                    p += a[i][j] * b[j][Observation[t+1]] * beta[t+1][j];
                beta[t][i] = p;
            }
        }
	}
	double p = 0;
	for(int j = 0; j < N; j++){
        p += PI[j] * b[j][Observation[0]] * beta[0][j];
	}
	return p;
}

double decode(int* o, int T, int* q)
{
    for (int t=0; t<T; ++t)
        for (int j=0; j<N; ++j)
            if (t == 0)
                δ[t][j] = π[j] * b[j][o[t]];
            else
            {
                double p = -1e9;
                for (int i=0; i<N; ++i)
                {
                    double w = δ[t-1][i] * a[i][j];
                    if (w > p) p = w, ψ[t][j] = i;
                }
                δ[t][j] = p * b[j][o[t]];
            }
 
    double p = -1e9;
    for (int j=0; j<N; ++j)
        if (δ[T-1][j] > p)
            p = δ[T-1][j], q[T-1] = j;
 
    for (int t=T-1; t>0; --t)
        q[t-1] = ψ[t][q[t]];
 
    return p;
}

double HMM(){
    cout << HMM_forward(T, Observation) << '\n';
    cout << HMM_backward(T, Observation) << '\n';
    cout << HMM_forward(T, &Observation[3]) + HMM_backward(T, &Observation[4])<< '\n';
}
