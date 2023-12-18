import torch

class Mechanical_Sound_Extraction():
    def __init__(self):
        pass

    def gcd(self, a, b):

        while b:
            a, b = b, a % b
        return a
    
    def ramanujan_prime_power_sum(self,p, alpha, n):

        if alpha == 1:
            return torch.tensor([(p - 1 if i % p == 0 else -1) for i in range(n)], dtype=torch.float)
        else:
            p_alpha = p ** alpha
            p_alpha_1 = p ** (alpha - 1)
            return torch.tensor([(0 if i % p_alpha_1 != 0 else (-p_alpha_1 if i % p_alpha != 0 else p_alpha_1 * (p - 1))) for i in range(n)], dtype=torch.float)

    def ramanujan_sum(self,Q):

        factors = self.prime_factors(Q)

        cq = torch.ones(Q, dtype=torch.float)

        for factor in set(factors):
            alpha = factors.count(factor)

            cq *= self.ramanujan_prime_power_sum(factor, alpha, Q)

        #cq = torch.where(torch.abs(cq) < 1e-6, torch.zeros_like(cq), cq)
        return cq

    def prime_factors(self,Q):
        n = Q.clone() if isinstance(Q, torch.Tensor) else Q

        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def Ramanujan_P(self,Q):
        
        cq = self.ramanujan_sum(Q)

        S_Q = torch.zeros((Q, Q),dtype=cq.dtype, device=Q.device)

        for i in range(Q):
            S_Q[i] = torch.roll(cq, shifts=i)

        return S_Q/Q

    def periodicity_measurement(self,X,N,Q):

        energry = torch.zeros((1,Q))

        E_xq = X.pow(2).sum()

        for i in range(1,Q+1):
            if N/Q <= 10:
                energry[i-1] = (N+i)*E_xq/(2*i)
            else:
                energry[i-1] = E_xq/(2*i)
                
        _, index_q = energy.max(0)

        return index_q+1

    def process_signal(self,X,Q,stop_condition):
        
        N = len(X)

        projected_result = troch.zeros_like(X)

        projected_save_last = troch.zeros_like(X)
        
        snr_list = []

        currt_snr = float("-inf")
        
        if currt_snr > stop_condition:
            
            q = self.periodicity_measurement(X,N,Q)

            M = math.ceil(N /q) 

            if N % q != 0:
                N_prime = q * M - N
                X = torch.cat([X, torch.zeros(N_prime).to(X.device)], dim=0)

            else:
                N_prime = 0

            X_prime = X.view(-1, q)

            block_sum = torch.sum(X_prime, axis=0)

            if N_prime > 0:
                block_mean = block_sum / M
                block_mean[-N_prime:] = block_sum[-N_prime:] / (M - 1)
            else:
                block_mean = block_sum / M

            R = self.Ramanujan_P(q)

            projected = torch.matmul(block_mean,R.to(X.device))

            projected_repeated = projected.repeat(M)[:N]

            X = X - projected_repeated

            projected_result +=  projected_repeated

            E_xq = X.pow(2).sum()

            E_projected_result = projected_result.pow(2).sum()

            E_r = E_xq - E_projected_result
            
            currt_snr = 10*torch.log10(E_projected_result/E_r)

            snr_list.append(currt_snr)

            projected_save_last = projected_result - projected_repeated
        
        else:
            return snr_list, projected_result, projected_save_last

    def Eliminate_SNR_errors(self,stop_condition,snr_list,projected_result,projected_save_last,M_):
        
        if  stop_condition <= ( snr_list[-2] + (snr_list[-1]-snr_list[-2])/M_):

            return projected_save_last
        else:

            return projected_result
