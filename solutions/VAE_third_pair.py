# ── Answer 1: add a third co-evolving pair (2, 25) ──────────────────────────

# Step 1: rebuild the MSA with the new grammar rule
coevolving_pairs_ex1 = [(4, 14), (9, 19), (2, 25)]   # added (2, 25)

def generate_sequence_ex1(rng):
    seq = rng.integers(0, V, size=L)
    for i, j in coevolving_pairs_ex1:
        seq[j] = seq[i]
    for pos in functional_sites:
        seq[pos] = rng.choice(allowed_at_functional[pos])
    return seq

rng_ex1 = np.random.default_rng(42)
msa_ex1 = np.array([generate_sequence_ex1(rng_ex1) for _ in range(N_train)])
print("Match rates:")
for i, j in coevolving_pairs_ex1:
    print(f"  ({i},{j}): {(msa_ex1[:,i]==msa_ex1[:,j]).mean():.2f}")

# Step 2: re-encode and retrain
X_oh_ex1  = torch.tensor(one_hot(msa_ex1, V))
X_idx_ex1 = torch.tensor(msa_ex1, dtype=torch.long)
loader_ex1 = DataLoader(TensorDataset(X_oh_ex1, X_idx_ex1), batch_size=256, shuffle=True)

model_ex1     = ProteinVAE(seq_len=L, vocab_size=V, latent_dim=2, hidden_dim=256).to(device)
optimizer_ex1 = torch.optim.Adam(model_ex1.parameters(), lr=1e-3)

for epoch in range(1, 81):
    model_ex1.train()
    for x_oh, x_idx in loader_ex1:
        x_oh, x_idx = x_oh.to(device), x_idx.to(device)
        logits, mu, log_var, _ = model_ex1(x_oh)
        loss = compute_elbo_loss(logits, x_idx, mu, log_var)
        optimizer_ex1.zero_grad(); loss.backward(); optimizer_ex1.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | loss: {loss.item():.3f}")

# Step 3: visualise — with 3 pairs but only 2 latent dims, the model can't fully disentangle all three.
# Two pairs will dominate (whichever carry the most information / reconstruction pressure).
# The third will appear as a weaker sub-structure within the dominant clusters.
model_ex1.eval()
with torch.no_grad():
    mu_ex1 = model_ex1.encode(X_oh_ex1.to(device))[0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (pos, title) in zip(axes, [
    (4,  'Co-evolving pos 4 (4,14)'),
    (9,  'Co-evolving pos 9 (9,19)'),
    (2,  'Co-evolving pos 2 (2,25) — NEW'),
]):
    idx = rng.choice(N_train, 2000, replace=False)
    sc = ax.scatter(mu_ex1[idx,0], mu_ex1[idx,1], c=msa_ex1[idx,pos],
                    cmap='tab10', vmin=0, vmax=V-1, alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax)
    ax.set_title(title, fontsize=9); ax.set_xlabel('z₁'); ax.set_ylabel('z₂')
plt.suptitle('Exercise 1: 3 co-evolving pairs, 2 latent dims\n'
             'Two pairs disentangle cleanly; the third is compressed into residual structure', fontsize=10)
plt.tight_layout(); plt.show()
