# Free Energy: Binary Symmetric Case

## Assumptions

1. **Binary tokens**: $x_t \in \{0, 1\}$.
2. **Symmetric emissions**: $e_A(0) = e_B(1) = a$, $e_A(1) = e_B(0) = 1 - a$, with $a > \frac{1}{2}$ (each state prefers one token).
3. **Symmetric prior**: $\pi_A = \pi_B = \frac{1}{2}$, so $h_0 = 0$.

---

## The field simplifies to a single value

$$h_t = \frac{1}{2}\log\frac{e_A(x_t)}{e_B(x_t)} = \begin{cases} +h & \text{if } x_t = 0, \\ -h & \text{if } x_t = 1, \end{cases}$$

where

$$h := \frac{1}{2}\log\frac{a}{1-a}.$$

So $h_t = h(1 - 2x_t)$, and the partial evidence sum becomes

$$H_\tau = \sum_{t=1}^{\tau} h_t = h\sum_{t=1}^{\tau}(1 - 2x_t) = h(\tau - 2N_\tau),$$

where $N_\tau = \sum_{t=1}^{\tau} x_t$ is the number of ones in the first $\tau$ tokens. Equivalently,

$$H_\tau = h(N_\tau^{(0)} - N_\tau^{(1)}),$$

where $N_\tau^{(0)} = \tau - N_\tau$ is the number of zeros.

---

## Frozen sector

With $h_0 = 0$:

$$Z_0 = e^{J(T-1)} \cdot 2\cosh(H_T) = e^{J(T-1)} \cdot 2\cosh\!\big(h(T - 2N_T)\big).$$

---

## Single-wall correction

$$\frac{Z_1}{Z_0} = y \sum_{\tau=1}^{T-1} \frac{\cosh(2H_\tau - H_T)}{\cosh(H_T)}.$$

Now $2H_\tau - H_T = 2h(\tau - 2N_\tau) - h(T - 2N_T) = h(2\tau - 4N_\tau - T + 2N_T)$.

Define the **magnetization** of the first $\tau$ tokens:

$$m_\tau := \tau - 2N_\tau = N_\tau^{(0)} - N_\tau^{(1)}.$$

Then $H_\tau = h \, m_\tau$, $H_T = h \, m_T$, and

$$2H_\tau - H_T = h(2m_\tau - m_T).$$

So the free energy is

$$\boxed{-F = J(T-1) + \log 2 + \log\cosh(h \, m_T) + y\sum_{\tau=1}^{T-1} \frac{\cosh\!\big(h(2m_\tau - m_T)\big)}{\cosh(h \, m_T)} + O(y^2),}$$

where everything is expressed in terms of:
- $J = \frac{1}{2}\log\frac{1-\varepsilon}{\varepsilon}$ (coupling),
- $h = \frac{1}{2}\log\frac{a}{1-a}$ (field strength),
- $y = \varepsilon/(1-\varepsilon)$ (wall fugacity),
- $m_\tau = \#\text{zeros} - \#\text{ones in first } \tau \text{ tokens}$ (running magnetization).

---

## Further simplification using $\cosh$ ratio identity

The ratio $\frac{\cosh(h(2m_\tau - m_T))}{\cosh(h \, m_T)}$ can be expanded. Write $\delta_\tau := 2m_\tau - m_T = 2(m_\tau - m_T) + m_T$, so:

$$\frac{\cosh(h\,\delta_\tau)}{\cosh(h\,m_T)} = \frac{e^{h\delta_\tau} + e^{-h\delta_\tau}}{e^{h m_T} + e^{-h m_T}}.$$

When the total evidence is strong ($|h \, m_T| \gg 1$, i.e., the sequence is clearly dominated by one token), and if say $m_T > 0$ (more zeros than ones, favoring state $A$):

$$\frac{\cosh(h\,\delta_\tau)}{\cosh(h\,m_T)} \approx e^{h(\delta_\tau - m_T)} + e^{-h(\delta_\tau + m_T)} \cdot \frac{1}{2}.$$

The dominant term is $e^{-2h(m_T - m_\tau)}$, which decays exponentially in how much the evidence after $\tau$ favors the majority state. This suppresses walls in the interior of long homogeneous runs.

---

## Special case: all zeros ($N_T = 0$)

If the sequence is all zeros, then $m_\tau = \tau$, $m_T = T$, and

$$\frac{\cosh(h(2\tau - T))}{\cosh(hT)} \xrightarrow{hT \gg 1} e^{-2h(T - \tau)} + e^{-2h\tau}.$$

So

$$\sum_{\tau=1}^{T-1} \frac{\cosh(h(2\tau - T))}{\cosh(hT)} \approx 2\sum_{\tau=1}^{T-1} e^{-2h\tau} \approx \frac{2}{e^{2h} - 1},$$

which is $O(1)$ independent of $T$. The wall correction to the free energy is just $\sim 2y/(e^{2h} - 1)$: a finite, $T$-independent shift. Walls are exponentially suppressed in a homogeneous sequence, as expected.

---

## Special case: half zeros, half ones ($m_T = 0$)

If $N_T = T/2$ (balanced sequence), then $m_T = 0$ and $\cosh(h \, m_T) = 1$, so

$$-F = J(T-1) + \log 2 + y\sum_{\tau=1}^{T-1} \cosh(2h \, m_\tau) + O(y^2).$$

Now the wall correction depends on the **detailed arrangement** of tokens through the running magnetization $m_\tau$. A balanced but alternating sequence (0101...) has $m_\tau \in \{0, 1\}$ for all $\tau$, giving a small correction $\sim Ty\cosh(2h)$. A balanced but clustered sequence (000...111...) has $m_\tau$ reaching $\pm T/2$, giving an exponentially large correction — signaling that a wall in the middle is strongly favored.
