class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=768):
        super(NWJ, self).__init__()        
            self.F_func = nn.Sequential(            
                nn.Linear(x_dim + y_dim, hidden_size),            
                nn.ReLU(),            
                nn.Linear(hidden_size, 1)        
                )        
    def forward(self, x_samples, y_samples):        # shuffle and concatenate        
        sample_size = y_samples.shape[0]                
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))        
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))                
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))        # shape [sample_size, sample_size, 1]        
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.                
        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()        
        return lower_bound

l_embeds = []
v_embeds = []
l_embeds_2 = []
v_embeds_2 = []
for i in range(input_mask.shape[0]):    
    l_embed = outputs["l_embeds"][-1][i][input_mask[i] > 0]    
    v_embed = outputs["v_embeds"][-1][i]    
    sim_score = F.normalize(l_embed, dim=-1) @ F.normalize(v_embed, dim=-1).transpose(0, 1)    # for L    
    l_embeds.append(l_embed)    
    v_embeds.append(v_embed[sim_score.max(1)[1], :])    # For V    
    l_embeds_2.append(l_embed[sim_score.max(0)[1], :])    
    v_embeds_2.append(v_embed)l_embeds = torch.cat(l_embeds + l_embeds_2, dim=0)
    v_embeds = torch.cat(v_embeds + v_embeds_2, dim=0)