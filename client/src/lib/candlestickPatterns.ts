export type Candle = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type PatternName =
  | "Doji"
  | "Hammer"
  | "Shooting Star"
  | "Bullish Engulfing"
  | "Bearish Engulfing";

export type DetectedPattern = {
  index: number; // candle index in array
  name: PatternName;
  direction: "bullish" | "bearish" | "neutral";
  strength: number; // 0-100 score
  rationale: string; // short explanation
  date: string; // candle date
};

const near = (a: number, b: number, tolerance: number) => Math.abs(a - b) <= tolerance;

// Helper to compute local trend (last 5 candles)
function getLocalTrend(candles: Candle[], index: number): number {
  if (index < 5) return 0;
  const slice = candles.slice(index - 5, index);
  const avgClose = slice.reduce((sum, c) => sum + c.close, 0) / slice.length;
  const currentClose = candles[index].close;
  return currentClose > avgClose ? 1 : currentClose < avgClose ? -1 : 0;
}

// Helper to compute strength based on context
function computeStrength(
  candles: Candle[],
  index: number,
  patternName: PatternName,
  direction: "bullish" | "bearish" | "neutral",
  quality: number // 0-1 base quality from pattern detection
): { strength: number; rationale: string } {
  const c = candles[index];
  const prev = candles[index - 1];
  const next = candles[index + 1];
  const trend = getLocalTrend(candles, index);
  
  let strength = quality * 60; // base 0-60 from pattern quality
  const factors: string[] = [];
  
  // Trend alignment bonus
  if (direction === "bullish" && trend > 0) {
    strength += 15;
    factors.push("aligns with uptrend");
  } else if (direction === "bearish" && trend < 0) {
    strength += 15;
    factors.push("aligns with downtrend");
  } else if (direction !== "neutral" && trend === 0) {
    strength -= 10;
    factors.push("weak trend context");
  }
  
  // Volume confirmation (if available)
  if (c.volume && prev?.volume) {
    const volRatio = c.volume / prev.volume;
    if (volRatio > 1.2) {
      strength += 10;
      factors.push("high volume");
    } else if (volRatio < 0.8) {
      strength -= 5;
      factors.push("low volume");
    }
  }
  
  // Next candle confirmation (if available)
  if (next) {
    if (direction === "bullish" && next.close > c.close) {
      strength += 10;
      factors.push("followed by rise");
    } else if (direction === "bearish" && next.close < c.close) {
      strength += 10;
      factors.push("followed by fall");
    }
  }
  
  strength = Math.max(0, Math.min(100, strength));
  const rationale = factors.length > 0 ? factors.join(", ") : "standard detection";
  
  return { strength: Math.round(strength), rationale };
}

export function detectPatterns(candles: Candle[]): DetectedPattern[] {
  const results: DetectedPattern[] = [];
  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    const prev = candles[i - 1];
    const range = Math.max(1e-6, c.high - c.low);
    const body = Math.abs(c.close - c.open);
    const upperWick = c.high - Math.max(c.open, c.close);
    const lowerWick = Math.min(c.open, c.close) - c.low;
    const bodyRatio = body / range;

    // Doji: very small body relative to range
    if (bodyRatio < 0.1) {
      const quality = 1 - bodyRatio * 10; // closer to 0 = stronger doji
      const { strength, rationale } = computeStrength(candles, i, "Doji", "neutral", quality);
      results.push({ 
        index: i, 
        name: "Doji", 
        direction: "neutral",
        strength,
        rationale,
        date: c.date,
      });
    }

    // Hammer: small body near top, long lower shadow
    const hammerLowerRatio = lowerWick / range;
    const hammerUpperRatio = upperWick / range;
    if (hammerLowerRatio > 0.5 && hammerUpperRatio < 0.2 && c.close > c.open) {
      const quality = Math.min(1, (hammerLowerRatio - 0.5) * 2); // longer shadow = stronger
      const { strength, rationale } = computeStrength(candles, i, "Hammer", "bullish", quality);
      results.push({ 
        index: i, 
        name: "Hammer", 
        direction: "bullish",
        strength,
        rationale,
        date: c.date,
      });
    }

    // Shooting Star: small body near bottom, long upper shadow
    if (upperWick / range > 0.5 && lowerWick / range < 0.2 && c.close < c.open) {
      const quality = Math.min(1, (upperWick / range - 0.5) * 2);
      const { strength, rationale } = computeStrength(candles, i, "Shooting Star", "bearish", quality);
      results.push({ 
        index: i, 
        name: "Shooting Star", 
        direction: "bearish",
        strength,
        rationale,
        date: c.date,
      });
    }

    // Engulfing patterns require previous candle
    if (prev) {
      const prevBodyTop = Math.max(prev.open, prev.close);
      const prevBodyBottom = Math.min(prev.open, prev.close);
      const currBodyTop = Math.max(c.open, c.close);
      const currBodyBottom = Math.min(c.open, c.close);
      const prevBodySize = prevBodyTop - prevBodyBottom;
      const currBodySize = currBodyTop - currBodyBottom;

      // Bullish engulfing: current up candle engulfs previous down body
      if (
        prev.close < prev.open &&
        c.close > c.open &&
        currBodyTop >= prevBodyTop &&
        currBodyBottom <= prevBodyBottom
      ) {
        const engulfRatio = currBodySize / Math.max(prevBodySize, 0.0001);
        const quality = Math.min(1, engulfRatio / 1.5); // larger engulf = stronger
        const { strength, rationale } = computeStrength(candles, i, "Bullish Engulfing", "bullish", quality);
        results.push({ 
          index: i, 
          name: "Bullish Engulfing", 
          direction: "bullish",
          strength,
          rationale,
          date: c.date,
        });
      }

      // Bearish engulfing: current down candle engulfs previous up body
      if (
        prev.close > prev.open &&
        c.close < c.open &&
        currBodyTop >= prevBodyTop &&
        currBodyBottom <= prevBodyBottom
      ) {
        const engulfRatio = currBodySize / Math.max(prevBodySize, 0.0001);
        const quality = Math.min(1, engulfRatio / 1.5);
        const { strength, rationale } = computeStrength(candles, i, "Bearish Engulfing", "bearish", quality);
        results.push({ 
          index: i, 
          name: "Bearish Engulfing", 
          direction: "bearish",
          strength,
          rationale,
          date: c.date,
        });
      }
    }
  }
  return results;
}


