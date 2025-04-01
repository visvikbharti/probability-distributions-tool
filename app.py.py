import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from math import factorial, exp, sqrt, pi

st.set_page_config(page_title="Probability Distributions Teaching Tool", layout="wide")

# Custom styling
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1 {text-align: center;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap;}
</style>
""", unsafe_allow_html=True)

st.title("Probability Distributions Teaching Tool")
st.markdown("### A visual guide to Binomial, Poisson, and Normal distributions")

# Create tabs for different sections
tabs = st.tabs(["Introduction", "Poisson Distribution", "Normal Distribution", "Binomial Approximations", "Real-world Applications"])

# Introduction tab
with tabs[0]:
    st.header("Introduction to Probability Distributions")
    
    st.subheader("What this tool demonstrates")
    st.markdown("""
    This interactive tool helps visualize and understand three key probability distributions:
    
    - **Binomial Distribution**: Models the number of successes in a fixed number of independent trials
    - **Poisson Distribution**: Models the number of events occurring in a fixed interval
    - **Normal Distribution**: Models continuous variables where values near the mean are most likely
    
    It also demonstrates how the Poisson and Normal distributions can approximate the Binomial distribution under certain conditions.
    
    Use the tabs above to explore each distribution and their relationships.
    """)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Probability_distributions_as_integrations_of_pdfs.png", 
                 caption="Probability distributions visualized", width=400)

# Poisson Distribution tab
with tabs[1]:
    st.header("The Poisson Distribution")
    
    st.markdown("""
    ### Mathematical Definition
    
    The Poisson distribution models the number of events occurring in a fixed interval of time or space, 
    given that these events occur independently and at a constant average rate.
    
    **Probability Mass Function (PMF):**
    
    $$P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$$
    
    Where:
    - $\\lambda$ (lambda) is the average number of events per interval
    - $k$ is the number of events (0, 1, 2, ...)
    - $e$ is Euler's number (≈ 2.71828)
    """)
    
    # Interactive Poisson distribution
    st.subheader("Interactive Poisson Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        lambda_val = st.slider("λ (lambda) parameter:", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
        max_k = st.slider("Maximum k value to display:", min_value=10, max_value=50, value=20)
        
        st.markdown("### Properties:")
        st.markdown(f"""
        - **Mean (μ):** {lambda_val}
        - **Variance:** {lambda_val}
        - **Standard Deviation:** {round(np.sqrt(lambda_val), 3)}
        - **Skewness:** {round(1/np.sqrt(lambda_val), 3)} (for λ > 0)
        - **Kurtosis:** {round(1/lambda_val, 3)} (excess kurtosis)
        """)
        
        # Show specific probabilities
        k_value = st.number_input("Calculate P(X = k) for k =", min_value=0, max_value=100, value=3)
        poisson_prob = stats.poisson.pmf(k_value, lambda_val)
        st.markdown(f"**P(X = {k_value}) = {poisson_prob:.6f}**")
        
        # Cumulative probability
        k_cum = st.number_input("Calculate P(X ≤ k) for k =", min_value=0, max_value=100, value=7)
        poisson_cum = stats.poisson.cdf(k_cum, lambda_val)
        st.markdown(f"**P(X ≤ {k_cum}) = {poisson_cum:.6f}**")
    
    with col2:
        # PMF plot
        fig, ax = plt.subplots(figsize=(10, 5))
        k_values = np.arange(0, max_k+1)
        poisson_pmf = stats.poisson.pmf(k_values, lambda_val)
        
        ax.bar(k_values, poisson_pmf, alpha=0.7)
        ax.set_xlabel('k (number of events)')
        ax.set_ylabel('Probability P(X = k)')
        ax.set_title(f'Poisson Distribution PMF (λ = {lambda_val})')
        ax.grid(True, alpha=0.3)
        
        # Add exact values as text above bars for smaller k values
        for i, v in enumerate(poisson_pmf):
            if i <= 10 and v > 0.01:  # Only annotate first few significant bars
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        st.pyplot(fig)
        
        # CDF plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        poisson_cdf = stats.poisson.cdf(k_values, lambda_val)
        
        ax2.step(k_values, poisson_cdf, where='post', lw=2)
        ax2.set_xlabel('k (number of events)')
        ax2.set_ylabel('Cumulative Probability P(X ≤ k)')
        ax2.set_title(f'Poisson Distribution CDF (λ = {lambda_val})')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
    
    st.subheader("Historical Context and Applications")
    st.markdown("""
    **Historical Development:**
    The Poisson distribution was introduced by French mathematician Siméon Denis Poisson in 1837 in his work 
    "Recherches sur la probabilité des jugements en matière criminelle et en matière civile" 
    (Research on the Probability of Judgments in Criminal and Civil Matters).
    
    **Key Applications:**
    - Count of rare events (e.g., number of earthquakes per year)
    - Number of arrivals (customers, calls, emails) in a time interval
    - Number of defects in a manufactured item
    - Number of mutations in a DNA sequence
    - Number of network failures in a time period
    - Insurance claims modeling
    """)

# Normal Distribution tab
with tabs[2]:
    st.header("The Normal Distribution")
    
    st.markdown("""
    ### Mathematical Definition
    
    The normal distribution is a continuous probability distribution that is symmetric about its mean.
    It is also known as the Gaussian distribution.
    
    **Probability Density Function (PDF):**
    
    $$f(x | \\mu, \\sigma^2) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}$$
    
    Where:
    - $\\mu$ (mu) is the mean
    - $\\sigma^2$ (sigma squared) is the variance
    - $\\sigma$ is the standard deviation
    """)
    
    # Interactive Normal distribution
    st.subheader("Interactive Normal Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        mu = st.slider("μ (mean):", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        sigma = st.slider("σ (standard deviation):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        st.markdown("### Properties:")
        st.markdown(f"""
        - **Mean:** {mu}
        - **Median:** {mu}
        - **Mode:** {mu}
        - **Variance:** {sigma**2}
        - **Skewness:** 0 (symmetric)
        - **Kurtosis:** 0 (excess kurtosis)
        """)
        
        # Area under the curve sliders
        st.subheader("Calculate probabilities")
        probability_type = st.radio("Calculation type:", 
                                   ["P(x ≤ X)", "P(X ≥ x)", "P(a ≤ X ≤ b)", "P(|X - μ| ≤ k)"])
        
        if probability_type == "P(x ≤ X)":
            x1 = st.slider("x value:", min_value=float(mu-4*sigma), max_value=float(mu+4*sigma), 
                          value=float(mu), step=0.1)
            prob = stats.norm.cdf(x1, loc=mu, scale=sigma)
            st.markdown(f"**P(X ≤ {x1}) = {prob:.6f}**")
            
        elif probability_type == "P(X ≥ x)":
            x1 = st.slider("x value:", min_value=float(mu-4*sigma), max_value=float(mu+4*sigma), 
                          value=float(mu), step=0.1)
            prob = 1 - stats.norm.cdf(x1, loc=mu, scale=sigma)
            st.markdown(f"**P(X ≥ {x1}) = {prob:.6f}**")
            
        elif probability_type == "P(a ≤ X ≤ b)":
            a, b = st.slider("Range [a, b]:", min_value=float(mu-4*sigma), max_value=float(mu+4*sigma), 
                            value=(float(mu-sigma), float(mu+sigma)), step=0.1)
            prob = stats.norm.cdf(b, loc=mu, scale=sigma) - stats.norm.cdf(a, loc=mu, scale=sigma)
            st.markdown(f"**P({a} ≤ X ≤ {b}) = {prob:.6f}**")
            
        elif probability_type == "P(|X - μ| ≤ k)":
            k = st.slider("k value (number of standard deviations):", 
                          min_value=0.1, max_value=4.0, value=1.0, step=0.1)
            prob = stats.norm.cdf(mu + k*sigma, loc=mu, scale=sigma) - stats.norm.cdf(mu - k*sigma, loc=mu, scale=sigma)
            st.markdown(f"**P(|X - μ| ≤ {k}σ) = {prob:.6f}**")
            
            # Show common rules
            if abs(k - 1.0) < 0.05:
                st.info("This is approximately the 68% rule (68.27% of data is within 1 standard deviation)")
            elif abs(k - 1.96) < 0.05:
                st.info("This is approximately the 95% rule (95% of data is within 1.96 standard deviations)")
            elif abs(k - 2.0) < 0.05:
                st.info("This is approximately the 95% rule (95.45% of data is within 2 standard deviations)")
            elif abs(k - 3.0) < 0.05:
                st.info("This is the 99.7% rule (99.73% of data is within 3 standard deviations)")
    
    with col2:
        # Create the PDF plot
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, pdf, 'b-', lw=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Normal Distribution PDF (μ = {mu}, σ = {sigma})')
        ax.grid(True, alpha=0.3)
        
        # Add shaded region based on selected probability
        if probability_type == "P(x ≤ X)":
            ax.fill_between(x, 0, pdf, where=(x >= x1), alpha=0.3, color='green')
            ax.axvline(x=x1, color='red', linestyle='--', alpha=0.7)
            ax.text(x1, max(pdf)/2, f'x = {x1}', rotation=90, va='center', ha='right')
            
        elif probability_type == "P(X ≥ x)":
            ax.fill_between(x, 0, pdf, where=(x <= x1), alpha=0.3, color='green')
            ax.axvline(x=x1, color='red', linestyle='--', alpha=0.7)
            ax.text(x1, max(pdf)/2, f'x = {x1}', rotation=90, va='center', ha='right')
            
        elif probability_type == "P(a ≤ X ≤ b)":
            ax.fill_between(x, 0, pdf, where=((x >= a) & (x <= b)), alpha=0.3, color='green')
            ax.axvline(x=a, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=b, color='red', linestyle='--', alpha=0.7)
            ax.text(a, max(pdf)/2, f'a = {a}', rotation=90, va='center', ha='right')
            ax.text(b, max(pdf)/2, f'b = {b}', rotation=90, va='center', ha='right')
            
        elif probability_type == "P(|X - μ| ≤ k)":
            ax.fill_between(x, 0, pdf, where=((x >= mu-k*sigma) & (x <= mu+k*sigma)), alpha=0.3, color='green')
            ax.axvline(x=mu-k*sigma, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=mu+k*sigma, color='red', linestyle='--', alpha=0.7)
            ax.text(mu-k*sigma, max(pdf)/2, f'μ-{k}σ', rotation=90, va='center', ha='right')
            ax.text(mu+k*sigma, max(pdf)/2, f'μ+{k}σ', rotation=90, va='center', ha='right')
        
        st.pyplot(fig)
        
        # Create the CDF plot
        cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(x, cdf, 'g-', lw=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f'Normal Distribution CDF (μ = {mu}, σ = {sigma})')
        ax2.grid(True, alpha=0.3)
        
        # Add markers for specific values on CDF
        if probability_type in ["P(x ≤ X)", "P(X ≥ x)"]:
            ax2.axvline(x=x1, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=stats.norm.cdf(x1, loc=mu, scale=sigma), color='red', linestyle='--', alpha=0.7)
            ax2.plot(x1, stats.norm.cdf(x1, loc=mu, scale=sigma), 'ro')
            
        elif probability_type == "P(a ≤ X ≤ b)":
            ax2.axvline(x=a, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=b, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=stats.norm.cdf(a, loc=mu, scale=sigma), color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=stats.norm.cdf(b, loc=mu, scale=sigma), color='red', linestyle='--', alpha=0.7)
            
        elif probability_type == "P(|X - μ| ≤ k)":
            ax2.axvline(x=mu-k*sigma, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=mu+k*sigma, color='red', linestyle='--', alpha=0.7)
            
        st.pyplot(fig2)
    
    st.subheader("Historical Context and Applications")
    st.markdown("""
    **Historical Development:**
    - Abraham de Moivre (1733): First derived as approximation to binomial
    - Pierre-Simon Laplace (1774): Used for error analysis
    - Carl Friedrich Gauss (1809): Formalized for astronomical observations (thus "Gaussian distribution")
    
    **Key Applications:**
    - Measurement errors in scientific experiments
    - Heights, weights, and other physical characteristics in populations
    - Test scores and educational measurements
    - Financial returns in markets
    - Quality control in manufacturing
    - Signal processing and noise modeling
    """)

# This code snippet should replace the existing Binomial Approximations tab section
# in your probability_distributions.py file

with tabs[3]:
    st.header("Binomial Approximations")
    
    st.markdown("""
    ### The Binomial Distribution
    
    The binomial distribution models the number of successes in a fixed number of independent trials, 
    each with the same probability of success.
    
    **Probability Mass Function (PMF):**
    
    $$P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}$$
    
    Where:
    - $n$ is the number of trials
    - $p$ is the probability of success in a single trial
    - $k$ is the number of successes (0, 1, 2, ..., n)
    """)
    
    st.markdown("""
    ### Approximations to the Binomial Distribution
    
    Under certain conditions, both the Poisson and Normal distributions can approximate the Binomial distribution:
    
    1. **Poisson Approximation**: When $n$ is large and $p$ is small, such that $\\lambda = np$ remains moderate.
       - Most effective when $n \\geq 20$, $p \\leq 0.05$, and $np < 10$
    
    2. **Normal Approximation**: When $n$ is large (regardless of $p$), with $\\mu = np$ and $\\sigma^2 = np(1-p)$.
       - Most effective when both $np \\geq 5$ and $n(1-p) \\geq 5$
    """)
    
    # Preset demonstrations with clear labels
    presets = st.radio(
        "Select a demonstration preset:",
        [
            "Normal approximation works well, Poisson does not (large n, moderate p)",
            "Poisson approximation works well, Normal does not (large n, very small p)",
            "Both approximations work well (large n, small p)",
            "Neither approximation works well (small n)",
            "Custom parameters"
        ],
        index=0
    )
    
    # Set parameters based on preset selection
    if presets == "Normal approximation works well, Poisson does not (large n, moderate p)":
        n_default = 100
        p_default = 0.3
        n_min, n_max = 50, 500
        p_min, p_max = 0.1, 0.5
    elif presets == "Poisson approximation works well, Normal does not (large n, very small p)":
        n_default = 1000
        p_default = 0.003
        n_min, n_max = 500, 2000
        p_min, p_max = 0.001, 0.01
    elif presets == "Both approximations work well (large n, small p)":
        n_default = 200
        p_default = 0.04
        n_min, n_max = 100, 500
        p_min, p_max = 0.01, 0.1
    elif presets == "Neither approximation works well (small n)":
        n_default = 10
        p_default = 0.3
        n_min, n_max = 5, 50
        p_min, p_max = 0.1, 0.5
    else:  # Custom parameters
        n_default = 100
        p_default = 0.1
        n_min, n_max = 5, 5000
        p_min, p_max = 0.001, 0.999
    
    # Interactive demonstration of approximations
    st.subheader("Interactive Demonstration of Approximations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Only show sliders if "Custom parameters" is selected
        if presets == "Custom parameters":
            n = st.slider("n (number of trials):", min_value=n_min, max_value=n_max, value=n_default, step=5)
            p = st.slider("p (probability of success):", min_value=p_min, max_value=p_max, value=p_default, step=0.001, format="%.3f")
        else:
            n = n_default
            p = p_default
            st.markdown(f"### Fixed Parameters:")
            st.markdown(f"- **n (number of trials): {n}**")
            st.markdown(f"- **p (probability of success): {p:.3f}**")
        
        # Derived parameters
        np_val = n * p
        npq = n * p * (1-p)
        
        st.markdown("### Derived Parameters:")
        st.markdown(f"""
        - **n·p = {np_val:.2f}**
        - **n·p·(1-p) = {npq:.2f}**
        """)
        
        # Suitability for approximations
        st.markdown("### Approximation Suitability:")
        
        # Poisson approximation
        if n >= 20 and p <= 0.05 and np_val < 10:
            poisson_suitable = "✅ Suitable"
            poisson_color = "green"
        else:
            poisson_suitable = "❌ Not ideal"
            poisson_color = "red"
        
        # Normal approximation
        if np_val >= 5 and n*(1-p) >= 5:
            normal_suitable = "✅ Suitable"
            normal_color = "green"
        else:
            normal_suitable = "❌ Not ideal"
            normal_color = "red"
        
        st.markdown(f"""
        - **Poisson Approximation:** <span style='color:{poisson_color}'>{poisson_suitable}</span>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        - **Normal Approximation:** <span style='color:{normal_color}'>{normal_suitable}</span>
        """, unsafe_allow_html=True)
        
        # Show suitability criteria
        with st.expander("Suitability Criteria"):
            st.markdown("""
            **Poisson Approximation Criteria:**
            - n ≥ 20 (large number of trials)
            - p ≤ 0.05 (small probability of success)
            - np < 10 (moderate expected number of successes)
            
            **Normal Approximation Criteria:**
            - np ≥ 5 (enough expected successes)
            - n(1-p) ≥ 5 (enough expected failures)
            """)
        
        # Error metrics
        st.markdown("### Approximation Errors:")
        
        # Create array of k values
        # Determine appropriate range around the mean
        mean = np_val
        std_dev = np.sqrt(npq)
        
        # Adjust range based on distribution shape
        if np_val < 10:
            # For small np values, include more of the left tail
            k_min = max(0, int(mean - 3*std_dev))
            k_max = min(n, int(mean + 4*std_dev))
        else:
            # For larger np values, use a more symmetric range
            k_min = max(0, int(mean - 4*std_dev))
            k_max = min(n, int(mean + 4*std_dev))
        
        k_vals = np.arange(k_min, k_max + 1)
        
        # Calculate exact binomial probabilities
        binom_pmf = stats.binom.pmf(k_vals, n, p)
        
        # Calculate Poisson approximation
        poisson_approx = stats.poisson.pmf(k_vals, np_val)
        
        # Calculate Normal approximation (with continuity correction)
        normal_approx = stats.norm.cdf(k_vals+0.5, loc=np_val, scale=np.sqrt(npq)) - \
                        stats.norm.cdf(k_vals-0.5, loc=np_val, scale=np.sqrt(npq))
        
        # Calculate total variation distance (TVD)
        tvd_poisson = 0.5 * np.sum(np.abs(binom_pmf - poisson_approx))
        tvd_normal = 0.5 * np.sum(np.abs(binom_pmf - normal_approx))
        
        # Kullback-Leibler divergence
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        kl_poisson = np.sum(binom_pmf * np.log((binom_pmf + epsilon) / (poisson_approx + epsilon)))
        kl_normal = np.sum(binom_pmf * np.log((binom_pmf + epsilon) / (normal_approx + epsilon)))
        
        st.markdown(f"""
        - **Poisson TVD Error:** <span style='color:{"green" if tvd_poisson < 0.1 else "orange" if tvd_poisson < 0.3 else "red"}'>{tvd_poisson:.4f}</span>
        - **Normal TVD Error:** <span style='color:{"green" if tvd_normal < 0.1 else "orange" if tvd_normal < 0.3 else "red"}'>{tvd_normal:.4f}</span>
        """, unsafe_allow_html=True)
        
        # Explain TVD
        st.markdown("""
        *TVD (Total Variation Distance) measures the maximum difference in probabilities. 
        Lower values indicate better approximation.*
        """)
        
        # Determine which approximation is better
        if tvd_poisson < tvd_normal:
            better_approx = "Poisson"
            difference = tvd_normal - tvd_poisson
        elif tvd_normal < tvd_poisson:
            better_approx = "Normal"
            difference = tvd_poisson - tvd_normal
        else:
            better_approx = "Both equally"
            difference = 0
        
        if difference > 0.05:
            st.success(f"**{better_approx} approximation is significantly better** (by {difference:.4f})")
    
    with col2:
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot exact binomial PMF
        ax.bar(k_vals, binom_pmf, alpha=0.5, label='Exact Binomial')
        
        # Plot Poisson approximation
        ax.plot(k_vals, poisson_approx, 'ro-', label='Poisson Approximation', alpha=0.7, markersize=3)
        
        # Plot Normal approximation
        ax.plot(k_vals, normal_approx, 'go-', label='Normal Approximation', alpha=0.7, markersize=3)
        
        ax.set_xlabel('k (number of successes)')
        ax.set_ylabel('Probability P(X = k)')
        ax.set_title(f'Binomial Distribution and Approximations (n={n}, p={p:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Create approximation error plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        # Calculate absolute errors
        poisson_errors = np.abs(binom_pmf - poisson_approx)
        normal_errors = np.abs(binom_pmf - normal_approx)
        
        # Plot errors with clearer differentiation
        ax2.bar(k_vals - 0.2, poisson_errors, width=0.4, color='red', alpha=0.7, label='Poisson Approximation Error')
        ax2.bar(k_vals + 0.2, normal_errors, width=0.4, color='green', alpha=0.7, label='Normal Approximation Error')
        
        ax2.set_xlabel('k (number of successes)')
        ax2.set_ylabel('Absolute Error |P_exact - P_approx|')
        ax2.set_title('Approximation Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for the largest errors
        poisson_max_error = np.max(poisson_errors)
        normal_max_error = np.max(normal_errors)
        
        poisson_max_k = k_vals[np.argmax(poisson_errors)]
        normal_max_k = k_vals[np.argmax(normal_errors)]
        
        ax2.annotate(f'Max: {poisson_max_error:.4f}',
                    xy=(poisson_max_k - 0.2, poisson_max_error),
                    xytext=(poisson_max_k - 0.2, poisson_max_error + 0.01),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    ha='center')
        
        ax2.annotate(f'Max: {normal_max_error:.4f}',
                    xy=(normal_max_k + 0.2, normal_max_error),
                    xytext=(normal_max_k + 0.2, normal_max_error + 0.01),
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    ha='center')
        
        st.pyplot(fig2)
    
    # Explanation of the theory behind the approximations
    st.subheader("Theoretical Underpinnings")
    
    col3, col4 = st.columns(2)
    
    with col3:
        with st.expander("Poisson Approximation to the Binomial"):
            st.markdown("""
            ### Poisson Approximation to the Binomial
            
            When $n \\to \\infty$, $p \\to 0$, and $np \\to \\lambda$ (a constant), the binomial distribution converges to the Poisson distribution with parameter $\\lambda$.
            
            Starting with the binomial PMF:
            
            $$P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}$$
            
            Substituting $p = \\lambda/n$:
            
            $$P(X = k) = \\binom{n}{k} \\left(\\frac{\\lambda}{n}\\right)^k \\left(1-\\frac{\\lambda}{n}\\right)^{n-k}$$
            
            Expanding the binomial coefficient:
            
            $$P(X = k) = \\frac{n!}{k!(n-k)!} \\frac{\\lambda^k}{n^k} \\left(1-\\frac{\\lambda}{n}\\right)^{n-k}$$
            
            Simplifying:
            
            $$P(X = k) = \\frac{n(n-1)\\ldots(n-k+1)}{k!} \\frac{\\lambda^k}{n^k} \\left(1-\\frac{\\lambda}{n}\\right)^{n} \\left(1-\\frac{\\lambda}{n}\\right)^{-k}$$
            
            As $n \\to \\infty$:
            - $\\frac{n(n-1)\\ldots(n-k+1)}{n^k} \\to 1$
            - $\\left(1-\\frac{\\lambda}{n}\\right)^{n} \\to e^{-\\lambda}$
            - $\\left(1-\\frac{\\lambda}{n}\\right)^{-k} \\to 1$
            
            Therefore:
            
            $$P(X = k) \\to \\frac{\\lambda^k e^{-\\lambda}}{k!}$$
            
            Which is the Poisson PMF with parameter $\\lambda$.
            """)
    
    with col4:
        with st.expander("Normal Approximation to the Binomial"):
            st.markdown("""
            ### Normal Approximation to the Binomial
            
            The De Moivre-Laplace theorem states that for large $n$, the binomial distribution can be approximated by a normal distribution with:
            
            - Mean $\\mu = np$
            - Variance $\\sigma^2 = np(1-p)$
            
            For a random variable $X \\sim \\text{Binomial}(n, p)$, the standardized random variable:
            
            $$Z = \\frac{X - np}{\\sqrt{np(1-p)}}$$
            
            approaches the standard normal distribution as $n \\to \\infty$.
            
            Since $X$ is discrete and the normal distribution is continuous, we apply a continuity correction:
            
            $$P(X = k) \\approx P(k-0.5 \\leq X \\leq k+0.5)$$
            
            $$P(X = k) \\approx \\Phi\\left(\\frac{k+0.5 - np}{\\sqrt{np(1-p)}}\\right) - \\Phi\\left(\\frac{k-0.5 - np}{\\sqrt{np(1-p)}}\\right)$$
            
            Where $\\Phi$ is the CDF of the standard normal distribution.
            """)
            
    # Practical guidelines
    st.subheader("Practical Guidelines for Choosing an Approximation")
    
    st.markdown("""
    ### When to Use Each Approximation:
    
    | Condition | Recommended Approximation |
    |-----------|---------------------------|
    | n is large, p is very small (≤ 0.01), np < 10 | **Poisson** is often better |
    | n is large, p is moderate (0.1 to 0.9) | **Normal** is better |
    | n is large, p is small but not tiny (0.01 to 0.05) | Either may work, check both |
    | Both np and n(1-p) are ≥ 10 | **Normal** is very accurate |
    | n < 20 | Use exact **Binomial** calculation |
    
    ### Historical Note:
    
    Before computers, these approximations were essential for practical calculations.
    Today, exact binomial calculations are feasible, but approximations remain valuable for:
    - Theoretical insights
    - Simplified calculations
    - Understanding limiting behavior
    """)

# Real-world Applications tab
with tabs[4]:
    st.header("Real-world Applications")
    
    st.markdown("""
    This section provides concrete examples of how these distributions are applied in various fields.
    
    Choose an example from the dropdown to see a simulation and analysis of that scenario.
    """)
    
    application = st.selectbox("Select an application to explore:", 
                              ["Email Arrivals (Poisson)",
                               "Quality Control (Normal)",
                               "Clinical Trials (Binomial and Normal Approximation)",
                               "Network Traffic (Poisson)",
                               "Manufacturing Defects (Binomial and Poisson Approximation)"])
    
    if application == "Email Arrivals (Poisson)":
        st.subheader("Email Arrivals - Poisson Process Simulation")
        
        st.markdown("""
        This example simulates email arrivals at a server during a workday. Emails arrive randomly 
        but at a consistent average rate, making this a perfect application of the Poisson distribution.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            hourly_rate = st.slider("Average email arrival rate (per hour):", 
                                  min_value=1, max_value=100, value=20)
            workday_hours = st.slider("Length of workday (hours):", 
                                     min_value=1, max_value=24, value=8)
            
            # Generate data
            np.random.seed(42)  # For reproducibility
            hours = np.arange(workday_hours + 1)
            
            # Simulate arrivals for each hour
            arrivals_per_hour = np.random.poisson(hourly_rate, workday_hours)
            total_arrivals = np.sum(arrivals_per_hour)
            
            # Calculate theoretical probabilities
            k_values = np.arange(0, workday_hours * hourly_rate * 2)
            daily_rate = hourly_rate * workday_hours
            poisson_pmf = stats.poisson.pmf(k_values, daily_rate)
            
            # Calculate probability of exceeding server capacity
            server_capacity = st.slider("Server capacity (emails/day):", 
                                      min_value=int(daily_rate/2), 
                                      max_value=int(daily_rate*2), 
                                      value=int(daily_rate*1.2))
            
            # Calculate probability of exceeding capacity
            exceed_prob = 1 - stats.poisson.cdf(server_capacity, daily_rate)
            
            st.markdown(f"""
            ### Analysis:
            - **Total expected emails per day:** {daily_rate}
            - **Simulated emails received:** {total_arrivals}
            - **Probability of exceeding capacity:** {exceed_prob:.4f} ({exceed_prob*100:.2f}%)
            """)
            
            if exceed_prob > 0.05:
                st.warning(f"There is a significant risk ({exceed_prob*100:.2f}%) of exceeding server capacity!")
            else:
                st.success(f"The risk of exceeding server capacity is low ({exceed_prob*100:.2f}%)")
        
        with col2:
            # Plot hourly arrivals
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Hourly arrivals plot
            ax1.bar(hours[:-1], arrivals_per_hour, width=0.7, alpha=0.7)
            ax1.axhline(y=hourly_rate, color='r', linestyle='--', alpha=0.7, label='Expected rate')
            ax1.set_xlabel('Hour of day')
            ax1.set_ylabel('Number of emails')
            ax1.set_title('Simulated email arrivals per hour')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily distribution
            ax2.bar(k_values, poisson_pmf, alpha=0.5, label='Poisson PMF')
            ax2.axvline(x=daily_rate, color='g', linestyle='--', alpha=0.7, label='Expected daily total')
            ax2.axvline(x=server_capacity, color='r', linestyle='--', alpha=0.7, label='Server capacity')
            ax2.axvline(x=total_arrivals, color='b', linestyle='--', alpha=0.7, label='Simulated total')
            
            # Shade the exceeded capacity region
            ax2.fill_between(k_values, 0, poisson_pmf, where=(k_values > server_capacity), 
                            color='red', alpha=0.3, label='Capacity exceeded')
            
            ax2.set_xlabel('Number of emails per day')
            ax2.set_ylabel('Probability')
            ax2.set_title('Daily email volume distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with st.expander("Application Details"):
            st.markdown("""
            ### Email Server Capacity Planning
            
            In IT operations, understanding the distribution of email arrivals is critical for:
            
            1. **Server capacity planning**: Ensuring the server can handle peak loads
            2. **Resource allocation**: Determining how many resources to allocate to email processing
            3. **SLA guarantees**: Predicting the probability of service disruptions
            
            The Poisson distribution is appropriate because:
            - Email arrivals are independent events
            - Arrivals occur at a roughly constant average rate
            - The probability of two emails arriving simultaneously is negligible
            
            In this simulation, we can use the probability of exceeding capacity to make informed decisions about server provisioning.
            """)
    
    elif application == "Quality Control (Normal)":
        st.subheader("Quality Control - Manufacturing Process")
        
        st.markdown("""
        This example simulates a manufacturing process where parts are produced with dimensions 
        that follow a normal distribution. We'll examine how quality control limits are set 
        to ensure specifications are met.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_dimension = st.slider("Target dimension (mm):", 
                                       min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            process_sigma = st.slider("Process standard deviation (mm):", 
                                   min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            
            # Set specification limits
            lower_spec = st.slider("Lower specification limit (mm):", 
                                 min_value=float(target_dimension-4*process_sigma), 
                                 max_value=float(target_dimension), 
                                 value=float(target_dimension-3*process_sigma), 
                                 step=0.01)
            
            upper_spec = st.slider("Upper specification limit (mm):", 
                                 min_value=float(target_dimension), 
                                 max_value=float(target_dimension+4*process_sigma), 
                                 value=float(target_dimension+3*process_sigma), 
                                 step=0.01)
            
            # Calculate process capability
            cp = (upper_spec - lower_spec) / (6 * process_sigma)
            
            # Calculate defect rates
            lower_defect_rate = stats.norm.cdf(lower_spec, loc=target_dimension, scale=process_sigma)
            upper_defect_rate = 1 - stats.norm.cdf(upper_spec, loc=target_dimension, scale=process_sigma)
            total_defect_rate = lower_defect_rate + upper_defect_rate
            
            # Calculate in PPM (parts per million)
            defect_ppm = total_defect_rate * 1_000_000
            
            st.markdown(f"""
            ### Process Analysis:
            - **Process Capability (Cp):** {cp:.3f}
            - **Total defect rate:** {total_defect_rate:.6f} ({defect_ppm:.1f} PPM)
            - **Lower spec defects:** {lower_defect_rate:.6f} ({lower_defect_rate*1_000_000:.1f} PPM)
            - **Upper spec defects:** {upper_defect_rate:.6f} ({upper_defect_rate*1_000_000:.1f} PPM)
            """)
            
            # Evaluate process capability
            if cp < 1.0:
                st.error(f"Process capability is inadequate (Cp < 1.0)")
            elif cp < 1.33:
                st.warning(f"Process capability is marginal (1.0 ≤ Cp < 1.33)")
            else:
                st.success(f"Process capability is adequate (Cp ≥ 1.33)")
        
        with col2:
            # Create control chart
            np.random.seed(123)
            samples = 50
            sample_size = 5
            
            # Generate sample data
            data = np.random.normal(target_dimension, process_sigma, samples * sample_size)
            data = data.reshape(samples, sample_size)
            
            # Calculate sample means and ranges
            sample_means = np.mean(data, axis=1)
            sample_ranges = np.max(data, axis=1) - np.min(data, axis=1)
            
            # Plot process distribution
            x = np.linspace(target_dimension - 4*process_sigma, target_dimension + 4*process_sigma, 1000)
            pdf = stats.norm.pdf(x, loc=target_dimension, scale=process_sigma)
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            
            # Process distribution plot
            axs[0].plot(x, pdf, 'b-', lw=2)
            axs[0].fill_between(x, 0, pdf, where=((x < lower_spec) | (x > upper_spec)), 
                              color='red', alpha=0.3, label='Out of spec')
            axs[0].axvline(x=lower_spec, color='r', linestyle='--', alpha=0.7, label='Spec limits')
            axs[0].axvline(x=upper_spec, color='r', linestyle='--', alpha=0.7)
            axs[0].axvline(x=target_dimension, color='g', linestyle='-', alpha=0.7, label='Target')
            axs[0].set_xlabel('Dimension (mm)')
            axs[0].set_ylabel('Probability Density')
            axs[0].set_title('Process Distribution and Specification Limits')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # Control chart
            axs[1].plot(range(1, samples+1), sample_means, 'bo-', alpha=0.7)
            axs[1].axhline(y=target_dimension, color='g', linestyle='-', label='Target')
            
            # Control limits (±3 sigma for means)
            mean_ucl = target_dimension + 3 * process_sigma / np.sqrt(sample_size)
            mean_lcl = target_dimension - 3 * process_sigma / np.sqrt(sample_size)
            
            axs[1].axhline(y=mean_ucl, color='r', linestyle='--', label='Control limits')
            axs[1].axhline(y=mean_lcl, color='r', linestyle='--')
            
            axs[1].set_xlabel('Sample Number')
            axs[1].set_ylabel('Sample Mean')
            axs[1].set_title('X-bar Control Chart')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with st.expander("Application Details"):
            st.markdown("""
            ### Manufacturing Quality Control
            
            Normal distributions are the foundation of Statistical Process Control (SPC) in manufacturing:
            
            1. **Process capability analysis**: Determining if a process can consistently produce parts within specifications
            2. **Control charts**: Monitoring process stability over time
            3. **Defect prediction**: Calculating expected defect rates
            
            The normal distribution is appropriate because:
            - Many manufacturing processes naturally produce variation that follows a normal distribution
            - The Central Limit Theorem ensures that sample means approach a normal distribution
            - Control limits are typically set at ±3 standard deviations from the mean
            
            **Process Capability Index (Cp)**
            
            Cp measures how well a process can produce output within specification limits:
            
            $C_p = \\frac{USL - LSL}{6\\sigma}$
            
            Where:
            - USL is the upper specification limit
            - LSL is the lower specification limit
            - σ is the process standard deviation
            
            Generally:
            - Cp < 1.0: Process is not capable
            - 1.0 ≤ Cp < 1.33: Process is marginally capable
            - Cp ≥ 1.33: Process is capable
            """)
            
    elif application == "Clinical Trials (Binomial and Normal Approximation)":
        st.subheader("Clinical Trials - Treatment Efficacy Analysis")
        
        st.markdown("""
        This example simulates a clinical trial where a new treatment is tested against a control group.
        We'll analyze the results using both the exact binomial distribution and its normal approximation.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Trial parameters
            sample_size = st.slider("Number of patients in each group:", 
                                  min_value=20, max_value=500, value=100, step=10)
            
            control_resp_rate = st.slider("Control group response rate:", 
                                        min_value=0.1, max_value=0.9, value=0.3, step=0.05)
            
            treatment_effect = st.slider("Treatment effect (increase in response rate):", 
                                       min_value=0.0, max_value=0.5, value=0.15, step=0.01)
            
            treatment_resp_rate = min(control_resp_rate + treatment_effect, 1.0)
            
            # Simulate trial
            np.random.seed(42)
            control_responses = np.random.binomial(1, control_resp_rate, sample_size)
            treatment_responses = np.random.binomial(1, treatment_resp_rate, sample_size)
            
            control_successes = np.sum(control_responses)
            treatment_successes = np.sum(treatment_responses)
            
            # Calculate statistics
            control_rate = control_successes / sample_size
            treatment_rate = treatment_successes / sample_size
            observed_diff = treatment_rate - control_rate
            
            # Test statistics
            # Exact test (Fisher's exact test)
            contingency_table = np.array([[treatment_successes, sample_size - treatment_successes],
                                        [control_successes, sample_size - control_successes]])
            
            odd_ratio, p_value_exact = stats.fisher_exact(contingency_table)
            
            # Normal approximation (Z-test for proportions)
            pooled_prop = (treatment_successes + control_successes) / (2 * sample_size)
            se = np.sqrt(pooled_prop * (1 - pooled_prop) * (2 / sample_size))
            z_stat = (treatment_rate - control_rate) / se
            p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            st.markdown(f"""
            ### Trial Results:
            - **Control group responses:** {control_successes}/{sample_size} ({control_rate:.1%})
            - **Treatment group responses:** {treatment_successes}/{sample_size} ({treatment_rate:.1%})
            - **Observed difference:** {observed_diff:.1%}
            
            ### Statistical Analysis:
            - **Exact Test p-value:** {p_value_exact:.4f}
            - **Normal Approximation p-value:** {p_value_normal:.4f}
            """)
            
            # Statistical conclusion
            alpha = 0.05
            if p_value_exact < alpha:
                st.success(f"The treatment shows a statistically significant effect (p < {alpha})")
            else:
                st.warning(f"The treatment effect is not statistically significant (p > {alpha})")
            
            # Compare the methods
            p_diff = abs(p_value_exact - p_value_normal)
            if p_diff > 0.01:
                st.info(f"Note: There is a substantial difference between exact and approximate p-values ({p_diff:.4f})")
        
        with col2:
            # Create plots for trial simulation
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            
            # Bar chart of results
            groups = ['Control', 'Treatment']
            response_rates = [control_rate, treatment_rate]
            
            axs[0].bar(groups, response_rates, color=['blue', 'green'], alpha=0.7)
            axs[0].set_ylim(0, 1.0)
            axs[0].set_ylabel('Response Rate')
            axs[0].set_title('Observed Response Rates')
            
            for i, v in enumerate(response_rates):
                axs[0].text(i, v + 0.02, f'{v:.1%}', ha='center')
            
            # Sampling distribution under null hypothesis
            k_values = np.arange(0, sample_size + 1)
            probs = stats.binom.pmf(k_values, sample_size, control_resp_rate)
            
            # Normal approximation to binomial
            x = np.linspace(0, sample_size, 1000)
            norm_approx = stats.norm.pdf(x, loc=sample_size*control_resp_rate, 
                                       scale=np.sqrt(sample_size*control_resp_rate*(1-control_resp_rate)))
            norm_approx = norm_approx * sample_size  # Scale to match binomial PMF
            
            axs[1].bar(k_values, probs, alpha=0.5, label='Exact Binomial')
            axs[1].plot(x, norm_approx, 'r-', lw=2, label='Normal Approximation')
            axs[1].axvline(x=control_successes, color='blue', linestyle='--', 
                         label=f'Control ({control_successes})')
            axs[1].axvline(x=treatment_successes, color='green', linestyle='--', 
                         label=f'Treatment ({treatment_successes})')
            
            axs[1].set_xlabel('Number of Responses (out of {})'.format(sample_size))
            axs[1].set_ylabel('Probability')
            axs[1].set_title('Sampling Distribution under Null Hypothesis')
            axs[1].legend()
            
            st.pyplot(fig)
        
        with st.expander("Application Details"):
            st.markdown("""
            ### Clinical Trial Statistical Analysis
            
            Clinical trials often use both exact and approximate methods for statistical analysis:
            
            1. **Exact Binomial Test**: Provides precise p-values but can be computationally intensive
            2. **Normal Approximation**: More computationally efficient, especially for large sample sizes
            
            **When to use the Normal Approximation:**
            
            The normal approximation to the binomial is appropriate when:
            - Sample sizes are large (generally n ≥ 30)
            - Both np ≥ 5 and n(1-p) ≥ 5
            
            **Power Analysis:**
            
            Power calculations for clinical trials often use the normal approximation to determine required sample sizes.
            
            For a two-sample proportion test with significance level α and power 1-β:
            
            $n = \\frac{(z_{1-\\alpha/2} + z_{1-\\beta})^2 \\cdot [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$
            
            Where:
            - p₁ and p₂ are the expected proportions in the two groups
            - z values are the quantiles from the standard normal distribution
            
            In practice, both exact and approximate methods are often reported, especially in regulatory submissions.
            """)
            
    elif application == "Network Traffic (Poisson)":
        st.subheader("Network Traffic - Queueing Theory Application")
        
        st.markdown("""
        This example demonstrates how the Poisson distribution is used in queueing theory to model
        network traffic and analyze server performance.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Queue parameters
            arrival_rate = st.slider("Arrival rate (requests/second):", 
                                   min_value=1.0, max_value=50.0, value=10.0, step=0.5)
            
            service_rate = st.slider("Service rate (requests/second):", 
                                   min_value=float(arrival_rate+0.5), max_value=100.0, 
                                   value=float(arrival_rate*1.5), step=0.5)
            
            simulation_time = st.slider("Simulation time (seconds):", 
                                     min_value=10, max_value=300, value=60, step=10)
            
            # Calculate M/M/1 queue metrics
            rho = arrival_rate / service_rate  # Utilization
            avg_queue_length = rho / (1 - rho)
            avg_waiting_time = 1 / (service_rate - arrival_rate)
            avg_system_time = 1 / (service_rate - arrival_rate) + 1 / service_rate
            
            # Probability of empty system
            p0 = 1 - rho
            
            # Probability of more than 10 customers in system
            p_more_than_10 = rho**11
            
            st.markdown(f"""
            ### M/M/1 Queue Metrics:
            - **Utilization (ρ):** {rho:.3f}
            - **Avg. queue length:** {avg_queue_length:.2f} requests
            - **Avg. waiting time:** {avg_waiting_time:.3f} seconds
            - **Avg. time in system:** {avg_system_time:.3f} seconds
            - **Probability of no requests:** {p0:.3f}
            - **Probability of >10 requests:** {p_more_than_10:.4f}
            """)
            
            if rho >= 0.9:
                st.error(f"System is highly loaded (ρ ≥ 0.9). Risk of instability is high!")
            elif rho >= 0.7:
                st.warning(f"System is moderately loaded (0.7 ≤ ρ < 0.9). Monitor performance.")
            else:
                st.success(f"System is lightly loaded (ρ < 0.7). Good performance expected.")
        
        with col2:
            # Simulate a Poisson process for arrivals
            np.random.seed(42)
            
            # Generate interarrival times (exponential distribution)
            interarrival_times = np.random.exponential(1/arrival_rate, size=int(arrival_rate*simulation_time*2))
            arrival_times = np.cumsum(interarrival_times)
            arrival_times = arrival_times[arrival_times <= simulation_time]
            
            # Generate service times (exponential distribution)
            service_times = np.random.exponential(1/service_rate, size=len(arrival_times))
            
            # Calculate departure times (simplified simulation)
            departure_times = np.zeros_like(arrival_times)
            system_times = np.zeros_like(arrival_times)
            queue_length = np.zeros(len(arrival_times) + 1)
            
            # Time when server becomes free
            server_free_time = 0
            
            for i in range(len(arrival_times)):
                # Customer enters service when they arrive or when server becomes free
                service_start = max(arrival_times[i], server_free_time)
                departure_times[i] = service_start + service_times[i]
                server_free_time = departure_times[i]
                
                # Calculate time in system
                system_times[i] = departure_times[i] - arrival_times[i]
                
                # Update queue length
                queue_length[i+1] = queue_length[i] + 1  # Arrival
                
                # Find all departures before next arrival
                if i < len(arrival_times) - 1:
                    next_arrival = arrival_times[i+1]
                    while i >= 0 and departure_times[i] <= next_arrival:
                        queue_length[i+1] -= 1  # Departure
                        i -= 1
            
            # Create plots
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            
            # Queue length over time
            time_points = np.concatenate(([0], arrival_times, departure_times))
            time_points.sort()
            
            # Reconstruct queue length at each time point
            ql_timepoints = np.zeros_like(time_points)
            current_queue = 0
            
            for i, t in enumerate(time_points):
                # Count arrivals
                arrivals = np.sum(arrival_times <= t)
                # Count departures
                departures = np.sum(departure_times <= t)
                ql_timepoints[i] = arrivals - departures
            
            axs[0].step(time_points, ql_timepoints, where='post', alpha=0.7)
            axs[0].axhline(y=avg_queue_length, color='r', linestyle='--', 
                         label=f'Theoretical avg: {avg_queue_length:.2f}')
            axs[0].set_xlabel('Time (seconds)')
            axs[0].set_ylabel('Queue Length')
            axs[0].set_title('Queue Length Over Time')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # System time histogram
            axs[1].hist(system_times, bins=20, alpha=0.7, density=True)
            
            # Theoretical system time distribution
            x = np.linspace(0, max(system_times)*1.5, 1000)
            theoretical_pdf = service_rate * (1 - rho) * np.exp(-service_rate * (1 - rho) * x)
            axs[1].plot(x, theoretical_pdf, 'r-', lw=2, 
                      label=f'Theoretical (avg: {avg_system_time:.3f}s)')
            
            axs[1].axvline(x=np.mean(system_times), color='b', linestyle='--', 
                         label=f'Simulated avg: {np.mean(system_times):.3f}s')
            axs[1].set_xlabel('Time in System (seconds)')
            axs[1].set_ylabel('Probability Density')
            axs[1].set_title('System Time Distribution')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with st.expander("Application Details"):
            st.markdown("""
            ### Queueing Theory and Network Traffic
            
            Queueing theory is fundamental to understanding network traffic and server performance:
            
            1. **M/M/1 Queue Model**: Single server with Poisson arrivals and exponential service times
            2. **Traffic Intensity (ρ)**: Ratio of arrival rate to service rate (ρ = λ/μ)
            3. **Performance Metrics**: Queue length, waiting time, system time
            
            **Key Assumptions**:
            - Arrivals follow a Poisson process (interarrival times are exponentially distributed)
            - Service times are exponentially distributed
            - FIFO service discipline
            - Infinite queue capacity
            
            **Theoretical Results for M/M/1 Queue**:
            
            - Mean number in system: $L = \\frac{\\rho}{1-\\rho}$
            - Mean time in system: $W = \\frac{1}{\\mu-\\lambda}$
            - Mean number in queue: $L_q = \\frac{\\rho^2}{1-\\rho}$
            - Mean waiting time in queue: $W_q = \\frac{\\rho}{\\mu-\\lambda}$
            
            **Applications**:
            - Traffic modeling in telecommunications
            - Web server capacity planning
            - Database transaction processing
            - Call center staffing
            
            In network engineering, these models help determine:
            - Required bandwidth for SLA compliance
            - Server capacity for expected traffic
            - Buffer sizes to prevent packet loss
            """)
    
    elif application == "Manufacturing Defects (Binomial and Poisson Approximation)":
        st.subheader("Manufacturing Defects - Quality Sampling")
        
        st.markdown("""
        This example simulates a quality control sampling process in manufacturing.
        We'll use both the binomial distribution and its Poisson approximation to analyze
        the probability of finding defects in sampled products.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Manufacturing parameters
            defect_rate = st.slider("Defect rate (proportion):", 
                                  min_value=0.001, max_value=0.1, value=0.01, step=0.001,
                                  format="%.3f")
            
            sample_size = st.slider("Sample size (items checked):", 
                                  min_value=10, max_value=500, value=100, step=10)
            
            # Maximum acceptable defects
            max_defects = st.slider("Maximum acceptable defects:", 
                                  min_value=0, max_value=10, value=2)
            
            # Calculate probabilities using binomial
            lambda_val = sample_size * defect_rate
            
            # Binomial probability of <= max_defects
            p_accept_binom = stats.binom.cdf(max_defects, sample_size, defect_rate)
            
            # Poisson approximation
            p_accept_poisson = stats.poisson.cdf(max_defects, lambda_val)
            
            # Calculate probability of exactly k defects for plotting
            k_values = np.arange(0, min(4*lambda_val + 10, sample_size) + 1)
            binom_pmf = stats.binom.pmf(k_values, sample_size, defect_rate)
            poisson_pmf = stats.poisson.pmf(k_values, lambda_val)
            
            # Calculate AOQ (Average Outgoing Quality)
            p_accept = p_accept_binom
            aoq = defect_rate * p_accept
            
            st.markdown(f"""
            ### Quality Control Analysis:
            - **Expected defects in sample:** {lambda_val:.2f}
            - **P(accept) - Binomial:** {p_accept_binom:.4f}
            - **P(accept) - Poisson approx.:** {p_accept_poisson:.4f}
            - **Average Outgoing Quality (AOQ):** {aoq:.6f}
            """)
            
            # Approximation error
            approx_error = abs(p_accept_binom - p_accept_poisson)
            st.markdown(f"**Approximation error:** {approx_error:.6f}")
            
            if approx_error > 0.01:
                st.warning(f"The Poisson approximation has significant error (> 0.01)")
            else:
                st.info(f"The Poisson approximation is accurate (error ≤ 0.01)")
                
            # Acceptance recommendation
            if p_accept_binom < 0.9:
                st.error(f"High rejection rate expected ({(1-p_accept_binom)*100:.1f}%)")
            elif p_accept_binom < 0.95:
                st.warning(f"Moderate rejection rate expected ({(1-p_accept_binom)*100:.1f}%)")
            else:
                st.success(f"Low rejection rate expected ({(1-p_accept_binom)*100:.1f}%)")
        
        with col2:
            # Create plots comparing binomial and Poisson
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            
            # PMF comparison
            bar_width = 0.4
            axs[0].bar(k_values - bar_width/2, binom_pmf, width=bar_width, 
                     alpha=0.7, label='Binomial PMF')
            axs[0].bar(k_values + bar_width/2, poisson_pmf, width=bar_width, 
                     alpha=0.7, label='Poisson PMF')
                     
            # Mark the acceptance region
            axs[0].axvline(x=max_defects + 0.5, color='r', linestyle='--', 
                         label=f'Max acceptable: {max_defects}')
            
            # Fill acceptance region
            axs[0].fill_between(k_values, 0, binom_pmf, where=(k_values <= max_defects), 
                              alpha=0.3, color='green', label='Acceptance region')
            
            axs[0].set_xlabel('Number of Defects')
            axs[0].set_ylabel('Probability')
            axs[0].set_title('Defect Distribution: Binomial vs Poisson Approximation')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # OC Curve (Operating Characteristic Curve)
            p_values = np.linspace(0.001, 0.05, 100)
            oc_binom = np.array([stats.binom.cdf(max_defects, sample_size, p) for p in p_values])
            oc_poisson = np.array([stats.poisson.cdf(max_defects, sample_size * p) for p in p_values])
            
            axs[1].plot(p_values, oc_binom, 'b-', lw=2, label='Binomial')
            axs[1].plot(p_values, oc_poisson, 'g--', lw=2, label='Poisson Approximation')
            axs[1].axvline(x=defect_rate, color='r', linestyle='--', 
                         label=f'Current defect rate: {defect_rate:.3f}')
            axs[1].axhline(y=0.1, color='orange', linestyle='-.', 
                         label='Producer\'s risk (10%)')
            
            axs[1].set_xlabel('Defect Rate (proportion)')
            axs[1].set_ylabel('Probability of Acceptance')
            axs[1].set_title('Operating Characteristic (OC) Curve')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with st.expander("Application Details"):
            st.markdown("""
            ### Quality Control Sampling in Manufacturing
            
            Statistical sampling is used to ensure product quality without inspecting every item:
            
            1. **Acceptance Sampling Plans**: Define sample size and acceptance criteria
            2. **Operating Characteristic (OC) Curve**: Shows probability of acceptance for different defect rates
            3. **Average Outgoing Quality (AOQ)**: Expected defect rate after inspection
            
            **Binomial Distribution**:
            
            The binomial distribution is exactly applicable when:
            - Each item is independently inspected
            - Each item is either defective or non-defective
            - The probability of a defect is constant
            
            **Poisson Approximation**:
            
            The Poisson approximation is appropriate when:
            - The sample size (n) is large
            - The defect rate (p) is small
            - Their product (np) is moderate
            
            This approximation simplifies calculations and is often used in industry standards.
            
            **Statistical Process Control Terms**:
            
            - **Producer's Risk (α)**: Probability of rejecting a good lot (Type I error)
            - **Consumer's Risk (β)**: Probability of accepting a bad lot (Type II error)
            - **Acceptable Quality Level (AQL)**: Defect level considered acceptable
            - **Lot Tolerance Percent Defective (LTPD)**: Defect level considered unacceptable
            
            Many industrial standards for sampling plans (e.g., ANSI/ASQ Z1.4, ISO 2859) are based on these statistical foundations.
            """)

# Add app run instructions
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
This interactive application demonstrates key probability distributions 
and their applications in various fields.

Use the tabs at the top to explore:
- **Introduction**: Overview of the distributions
- **Poisson Distribution**: Explore its properties and behavior
- **Normal Distribution**: Visualize and understand normal distribution properties
- **Binomial Approximations**: See how Poisson and Normal can approximate the Binomial
- **Real-world Applications**: Practical examples where these distributions are used
""")

st.sidebar.header("How to Run This App")
st.sidebar.markdown("""
1. Save this code to a file named `probability_distributions.py`
2. Make sure you have installed:
   ```
   pip install streamlit numpy matplotlib scipy pandas
   ```
3. Run the application with:
   ```
   streamlit run probability_distributions.py
   ```
4. The app will open in your default web browser
""")

st.sidebar.header("Teaching Tips")
st.sidebar.markdown("""
- Start with the intuitive understanding before diving into the mathematics
- Use the interactive elements to demonstrate parameter effects
- Connect theoretical concepts with real-world applications
- Compare the different distributions using the Binomial Approximations tab
- Show how approximations become more accurate under specific conditions
""")