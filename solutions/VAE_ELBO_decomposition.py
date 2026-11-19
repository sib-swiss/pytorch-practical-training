# ── Answer 2: decompose ΔELBO into reconstruction and KL terms ───────────────

def score_variants_decomposed(model, wildtype, variants, device, n_samples=50):
    """Returns ΔELBO, Δreconstruction, and ΔKL separately for each variant."""
    model.eval()
    all_seqs = np.vstack([wildtype[None], np.array(variants)])   # wildtype first, then variants
    oh  = torch.tensor(one_hot(all_seqs, V), device=device)
    idx = torch.tensor(all_seqs, dtype=torch.long, device=device)

    with torch.no_grad():
        mu, log_var = model.encode(oh)

        # Reconstruction term: E[log p(x|z)] estimated by Monte Carlo
        recon_sum = torch.zeros(oh.shape[0], device=device)
        for _ in range(n_samples):
            z      = model.reparameterise(mu, log_var)
            logits = model.decode(z)
            log_px_z = -F.cross_entropy(
                logits.view(-1, V), idx.view(-1), reduction='none'
            ).view(oh.shape[0], L).sum(dim=1)
            recon_sum += log_px_z
        recon = (recon_sum / n_samples).cpu().numpy()   # (N_variants+1,)

        # KL term: KL(q(z|x) || p(z))
        kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)).cpu().numpy()

    elbo = recon - kl

    # Compute deltas relative to wildtype (index 0)
    delta_elbo  = elbo[1:]  - elbo[0]
    delta_recon = recon[1:] - recon[0]
    delta_kl    = kl[1:]    - kl[0]   # positive ΔKL = mutant pushes further from prior

    return delta_elbo, delta_recon, delta_kl


# Score grammar-breaking and neutral variants with decomposed terms
d_elbo_gb1, d_recon_gb1, d_kl_gb1       = score_variants_decomposed(model, wildtype, gb1_variants,     device)
d_elbo_gb2, d_recon_gb2, d_kl_gb2       = score_variants_decomposed(model, wildtype, gb2_variants,     device)
d_elbo_neu, d_recon_neu, d_kl_neu       = score_variants_decomposed(model, wildtype, neutral_variants, device)

# Plot the decomposition
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, d_elbo, d_recon, d_kl, label, color in [
    (axes[0], d_elbo_gb1,  d_recon_gb1,  d_kl_gb1,  'Grammar-breaking - functional', 'firebrick'),
    (axes[1], d_elbo_gb2,  d_recon_gb2,  d_kl_gb2,  'Grammar-breaking - co-evolving', 'orange'),
    (axes[2], d_elbo_neu, d_recon_neu, d_kl_neu, 'Neutral',           'steelblue'),
]:
    x = np.arange(len(d_elbo))
    ax.bar(x,        d_recon, label='Δreconstruction', color='C0', alpha=0.8)
    ax.bar(x,       -d_kl,    label='−ΔKL',            color='C1', alpha=0.8, bottom=d_recon)
    ax.axhline(0, color='k', lw=1)
    ax.set_title(f'{label} variants\nΔELBO decomposed into reconstruction and KL', fontsize=9)
    ax.set_xlabel('Variant index'); ax.set_ylabel('ΔELBO contribution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.suptitle('Where does the low ELBO come from? Reconstruction failure or posterior shift?',
             fontsize=11, y=1.02)
plt.show()

print("\nInterpretation:")
print("  Grammar-breaking variants: low ΔELBO is driven mainly by the RECONSTRUCTION term.")
print("  The decoder can't reconstruct the sequence well because the latent code z")
print("  doesn't correspond to any region the model learned to decode.")
print("  → The evolutionary memory lives in the DECODER (p(x|z)), not the encoder.")
print()
print("  Neutral variants: both terms near 0 : the model encodes and decodes them without surprise.")
