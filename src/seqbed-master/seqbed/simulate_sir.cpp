#include <random>
#include <iostream>
#include <chrono>

void simulate_sir_cpp(
  int *S, int *I, int *R,
  int S0, int I0, int R0,
  float d,
  int N,
  float beta,
  float gamma
  )
{

  // std::cout << "S0: " << S0 << "\tI0: " << I0 << "\tR0: " << R0 << std::endl;
  // std::cout << "d: " << d << "\tN: " << N << std::endl;
  // std::cout << "beta: " << beta << "\tgamma: " << gamma << std::endl;

  // setup and seed generator
  std::default_random_engine generator;
  generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

  // init stuff
  float dt = 0.01;
  float this_d = 0;
  int St = S0;
  int It = I0;
  int Rt = R0;
  int dI;
  int dR;
  float pinf = 0.0;
  float precov = gamma;
  
  // simulate
  while (this_d < d) {
    this_d += dt;

    // deltas    
    pinf = beta * ((float) It) / ((float) N);
    std::binomial_distribution<int> dist_I(St, pinf);
    dI = dist_I(generator);
    std::binomial_distribution<int> dist_R(It, precov);
    dR = dist_R(generator);

    // update SIR
    St = St - dI;
    It = It + dI - dR;
    Rt = Rt + dR;
  }  

  // final values
  *S = St;
  *I = It;
  *R = Rt;
}
  
extern "C"
{
  void simulate_sir(int *S, int *I, int *R,
    int S0, int I0, int R0,
    float d,
    int N,
    float beta,
    float gamma
  )
  { return simulate_sir_cpp(S, I, R, S0, I0, R0, d, N, beta, gamma); }
}

/* 
LINUX:
g++ -c -fPIC simulate_sir.cpp -o simulate_sir.o
g++ -shared -o libsimulate_sir.so simulate_sir.o

WINDOWS:
g++ -c -fPIC simulate_sir.cpp -o simulate_sir.o
g++ -shared simulate_sir.o -o libsimulate_sir.dll -Wl,--out-implib,libsimulate_sir.a
*/