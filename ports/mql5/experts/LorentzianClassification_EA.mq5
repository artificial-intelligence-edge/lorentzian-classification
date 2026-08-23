//+------------------------------------------------------------------+
//|                                 LorentzianClassification_EA.mq5   |
//| Thin EA wrapper for the Lorentzian Classification indicator.     |
//|                                                                  |
//| Loads the indicator via iCustom and trades the signals it plots: |
//|   - opens on the indicator's Buy / Sell arrows (last closed bar) |
//|   - optionally closes on the indicator's Exit Buy / Exit Sell    |
//|     signals, and reverses on an opposite arrow                   |
//|   - optional fixed Stop Loss / Take Profit (in points)           |
//| All indicator signal parameters are exposed for optimization.    |
//+------------------------------------------------------------------+
#property copyright   "AI Edge"
#property link        "https://ai-edge.io/"
#property version     "1.01"
#property description "Thin EA wrapper for the Lorentzian Classification indicator."
#property description "Trades the indicator's Buy/Sell arrows and Exit signals; optional SL/TP."
// Embed the compiled indicator so the EA .ex5 is self-contained (fresh
// terminals need no separate indicator install). Resource names are capped
// at 63 chars, so the indicator is embedded from a flat path: compile the
// indicator first, copy its .ex5 to MQL5\Indicators\, then compile the EA.
#resource "\\Indicators\\LorentzianClassification.ex5"

#include <Trade/Trade.mqh>

// =====================================================================
// EA Settings
// =====================================================================
input  group "=== EA Settings ==="
input  double InpLotSize           = 0.1;    // Position size (lots)
sinput int    InpMagicNumber       = 77701;  // Unique EA magic number
input  int    InpStopLossPoints    = 0;      // Stop Loss (points; 0 = off)
input  int    InpTakeProfitPoints  = 0;      // Take Profit (points; 0 = off)
sinput int    InpSlippagePoints    = 10;     // Max slippage / deviation (points)
input  bool   InpUseIndicatorExits = false;  // Close on the indicator's Exit Buy/Sell signals
sinput int    InpMinTrades         = 30;     // Min trades for a valid pass (Custom-max Sortino gate)

// =====================================================================
// Indicator parameters (mirrored for iCustom pass-through)
// Order and names match LorentzianClassification.mq5 exactly.
// =====================================================================
input group "=== General Settings ==="
input ENUM_APPLIED_PRICE InpSource = PRICE_CLOSE; // Source: Source of the input data
input int    InpNeighborsCount   = 8;     // Neighbors Count: Number of neighbors to consider (1-100)
input int    InpMaxBarsBack      = 2000;  // Max Bars Back: Max historical bars used for ML lookups
input int    InpFeatureCount     = 5;     // Feature Count: Features used for ML predictions (2-5)
input int    InpColorCompression = 1;     // Color Compression: color intensity factor (1-10)
input bool   InpUseDynamicExits  = false; // Use Dynamic Exits: adjust exit threshold via kernel regression
input bool   InpIncludeFullHist  = false; // Include Full History: train ANN on all bars, not just recent
input bool   InpShowTradeStats   = true;  // Show Trade Stats: table for calibration only, NOT a backtest
input bool   InpUseWorstCase     = false; // Use Worst Case Estimates: close-only estimates in trade stats

input group "=== Filters ==="
input bool   InpUseVolFilter     = true;  // Use Volatility Filter: recentATR(1) > historicalATR(10)
input bool   InpUseRegimeFilter  = true;  // Use Regime Filter: Kaufman Adaptive Slope, trending vs ranging
input bool   InpUseAdxFilter     = false; // Use ADX Filter: require ADX > threshold for signal flips
input double InpRegimeThreshold  = -0.1;  // Regime Threshold: (-10 to 10); higher = require stronger trend
input int    InpAdxThreshold     = 20;    // ADX Threshold: (0-100); higher = require stronger trend
input bool   InpUseEmaFilter     = false; // Use EMA Filter: long only when close > EMA, short when below
input int    InpEmaPeriod        = 200;   // EMA Period: Period of the EMA used for the EMA filter
input bool   InpUseSmaFilter     = false; // Use SMA Filter: long only when close > SMA, short when below
input int    InpSmaPeriod        = 200;   // SMA Period: Period of the SMA used for the SMA filter

input group "=== Kernel Settings ==="
input bool   InpUseKernelFilter    = true;  // Trade with Kernel: gate entries by kernel direction
input bool   InpShowKernelEst      = true;  // Show Kernel Estimate: plot kernel regression on the chart
input bool   InpUseKernelSmoothing = false; // Enhance Kernel Smoothing: fewer color flips, more ML entries
input int    InpKernelH            = 8;     // Lookback Window: (3-50); sliding window of recent bars
input double InpKernelR            = 8.0;   // Relative Weighting: ->0 favors long timeframes; ->inf Gaussian
input int    InpKernelX            = 25;    // Regression Level: (2-25); smaller=tighter, larger=looser fit
input int    InpKernelLag          = 2;     // Lag: crossover detection (1-2); lower = earlier crossovers

input group "=== Feature Engineering ==="
input int    InpF1Type   = 0;    // Feature 1: (0=RSI, 1=WT, 2=CCI, 3=ADX)
input int    InpF1ParamA = 14;   // Parameter A: Primary parameter of feature 1
input int    InpF1ParamB = 1;    // Parameter B: Secondary parameter of feature 1 (if applicable)
input int    InpF2Type   = 1;    // Feature 2: (0=RSI, 1=WT, 2=CCI, 3=ADX)
input int    InpF2ParamA = 10;   // Parameter A: Primary parameter of feature 2
input int    InpF2ParamB = 11;   // Parameter B: Secondary parameter of feature 2 (if applicable)
input int    InpF3Type   = 2;    // Feature 3: (0=RSI, 1=WT, 2=CCI, 3=ADX)
input int    InpF3ParamA = 20;   // Parameter A: Primary parameter of feature 3
input int    InpF3ParamB = 1;    // Parameter B: Secondary parameter of feature 3 (if applicable)
input int    InpF4Type   = 3;    // Feature 4: (0=RSI, 1=WT, 2=CCI, 3=ADX)
input int    InpF4ParamA = 20;   // Parameter A: Primary parameter of feature 4
input int    InpF4ParamB = 2;    // Parameter B: Secondary parameter of feature 4 (if applicable)
input int    InpF5Type   = 0;    // Feature 5: (0=RSI, 1=WT, 2=CCI, 3=ADX)
input int    InpF5ParamA = 9;    // Parameter A: Primary parameter of feature 5
input int    InpF5ParamB = 1;    // Parameter B: Secondary parameter of feature 5 (if applicable)

input group "=== Display Settings ==="
input bool   InpShowBarColors = true;  // Show Bar Colors: Color each bar by the ML model's prediction
input bool   InpShowBarPreds  = true;  // Show Bar Prediction Values: integer prediction on each bar
input bool   InpUseAtrOffset  = false; // Use ATR Offset: ATR offset instead of prediction offset
input double InpBarPredOffset  = 0;     // Bar Prediction Offset: % offset from the bar high/low

input group "=== Alert Settings ==="
input bool   InpEnableSignalAlerts      = false; // Enable Signal Alerts: terminal alert on a confirmed Buy/Sell signal
input bool   InpEnablePushNotifications = false; // Enable Push Notifications: also send to configured MetaQuotes IDs

// =====================================================================
// Globals
// =====================================================================
const string IndicatorPath = "::Indicators\\LorentzianClassification.ex5";
int    g_indHandle = INVALID_HANDLE;
CTrade g_trade;

// Indicator buffer indices (must match the indicator's SetIndexBuffer order).
#define BUF_BUY        0   // Buy signal       (price on entry bar, else EMPTY_VALUE)
#define BUF_SELL       1   // Sell signal      (price on entry bar, else EMPTY_VALUE)
#define BUF_EXIT_BUY   2   // Exit-long signal (price on exit bar,  else EMPTY_VALUE)
#define BUF_EXIT_SELL  3   // Exit-short signal
// Indices 4-7 (Kernel, KernelColor, Direction, Prediction) and 8-12 (color
// candle OHLC + color) are produced by the indicator but not used by this EA.

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // ----- Validate inputs (user-fixable -> INIT_PARAMETERS_INCORRECT) -----
   if(InpLotSize <= 0.0)
   { Print("InpLotSize must be > 0."); return INIT_PARAMETERS_INCORRECT; }
   if(InpMagicNumber <= 0)
   { Print("InpMagicNumber must be > 0."); return INIT_PARAMETERS_INCORRECT; }
   if(InpStopLossPoints < 0 || InpTakeProfitPoints < 0 || InpSlippagePoints < 0)
   { Print("SL/TP/slippage points cannot be negative."); return INIT_PARAMETERS_INCORRECT; }
   if(InpNeighborsCount < 1)
   { Print("InpNeighborsCount must be >= 1."); return INIT_PARAMETERS_INCORRECT; }
   if(InpMaxBarsBack < 1)
   { Print("InpMaxBarsBack must be >= 1."); return INIT_PARAMETERS_INCORRECT; }
   if(InpFeatureCount < 2 || InpFeatureCount > 5)
   { Print("InpFeatureCount must be between 2 and 5."); return INIT_PARAMETERS_INCORRECT; }

   // ----- Create the indicator handle (once) -----
   // The "Show Exits" slot is driven by InpUseIndicatorExits so the indicator
   // populates its Exit buffers only when the EA actually consumes them.
   g_indHandle = iCustom(
      _Symbol, _Period, IndicatorPath,
      // General
      InpSource,
      InpNeighborsCount, InpMaxBarsBack, InpFeatureCount,
      InpColorCompression, InpUseIndicatorExits, InpUseDynamicExits,
      InpIncludeFullHist, InpShowTradeStats, InpUseWorstCase,
      // Filters
      InpUseVolFilter, InpUseRegimeFilter, InpUseAdxFilter,
      InpRegimeThreshold, InpAdxThreshold,
      InpUseEmaFilter, InpEmaPeriod, InpUseSmaFilter, InpSmaPeriod,
      // Kernel
      InpUseKernelFilter, InpShowKernelEst, InpUseKernelSmoothing,
      InpKernelH, InpKernelR, InpKernelX, InpKernelLag,
      // Features
      InpF1Type, InpF1ParamA, InpF1ParamB,
      InpF2Type, InpF2ParamA, InpF2ParamB,
      InpF3Type, InpF3ParamA, InpF3ParamB,
      InpF4Type, InpF4ParamA, InpF4ParamB,
      InpF5Type, InpF5ParamA, InpF5ParamB,
      // Display
      InpShowBarColors, InpShowBarPreds,
      InpUseAtrOffset, InpBarPredOffset,
      // Alerts (trailing inputs preserve the original positional mapping)
      InpEnableSignalAlerts, InpEnablePushNotifications
   );

   if(g_indHandle == INVALID_HANDLE)
   {
      Print("ERROR: could not load indicator '", IndicatorPath,
            "'. Error: ", GetLastError());
      return INIT_FAILED;
   }

   // ----- Configure the trade helper -----
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints((ulong)InpSlippagePoints);
   g_trade.SetMarginMode();
   g_trade.SetTypeFillingBySymbol(_Symbol); // pick a filling mode the symbol allows

   Print("LorentzianClassification_EA initialized. K=", InpNeighborsCount,
         " Bars=", InpMaxBarsBack, " Features=", InpFeatureCount,
         " | SL=", InpStopLossPoints, " TP=", InpTakeProfitPoints,
         " | IndicatorExits=", (InpUseIndicatorExits ? "on" : "off"));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Cleanup                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_indHandle != INVALID_HANDLE)
   {
      IndicatorRelease(g_indHandle);
      g_indHandle = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
//| Read one indicator buffer value at the last closed bar (index 1).|
//| Returns false if the data is not ready yet (copy failed); sets   |
//| 'active' true when the value is a real (non-empty) signal.        |
//+------------------------------------------------------------------+
bool ReadSignal(const int bufferIndex, bool &active)
{
   double v[1];
   active = false;
   if(CopyBuffer(g_indHandle, bufferIndex, 1, 1, v) != 1)
      return false; // not calculated yet -> caller should retry next tick
   active = (v[0] != EMPTY_VALUE && v[0] != 0.0);
   return true;
}

//+------------------------------------------------------------------+
//| Count this EA's open positions (by symbol + magic).              |
//+------------------------------------------------------------------+
void CountPositions(bool &haveBuy, bool &haveSell)
{
   haveBuy  = false;
   haveSell = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetTicket(i) == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      long posType = PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY)       haveBuy  = true;
      else if(posType == POSITION_TYPE_SELL) haveSell = true;
   }
}

//+------------------------------------------------------------------+
//| Normalize a lot size to the symbol's min/max/step.               |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
{
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(stepLot > 0.0)
      lot = MathFloor(lot / stepLot + 1e-9) * stepLot; // round DOWN to step (epsilon guards fp)
   if(lot < minLot) lot = minLot;                       // clamp AFTER stepping
   if(lot > maxLot) lot = maxLot;
   int lotDigits = (stepLot > 0.0) ? (int)MathMax(0.0, -MathLog10(stepLot)) : 2;
   return NormalizeDouble(lot, lotDigits);
}

//+------------------------------------------------------------------+
//| Close all of this EA's positions of the given type.              |
//| Returns true only if every matching position closed cleanly.     |
//+------------------------------------------------------------------+
bool ClosePositions(const long posType)
{
   bool allClosed = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetInteger(POSITION_TYPE) != posType) continue;

      bool ok = g_trade.PositionClose(ticket);
      uint rc = g_trade.ResultRetcode();
      if(!ok || (rc != TRADE_RETCODE_DONE && rc != TRADE_RETCODE_DONE_PARTIAL))
      {
         allClosed = false;
         Print("PositionClose failed: ticket=", ticket, " retcode=", rc,
               " (", g_trade.ResultRetcodeDescription(), ")");
      }
   }
   return allClosed;
}

//+------------------------------------------------------------------+
//| Open a market position with optional SL/TP (points).             |
//+------------------------------------------------------------------+
void OpenPosition(const ENUM_ORDER_TYPE type)
{
   double lot   = NormalizeLot(InpLotSize);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                           : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = 0.0, tp = 0.0;
   if(InpStopLossPoints > 0)
      sl = (type == ORDER_TYPE_BUY) ? price - InpStopLossPoints * point
                                    : price + InpStopLossPoints * point;
   if(InpTakeProfitPoints > 0)
      tp = (type == ORDER_TYPE_BUY) ? price + InpTakeProfitPoints * point
                                    : price - InpTakeProfitPoints * point;
   if(sl > 0.0) sl = NormalizeDouble(sl, _Digits);
   if(tp > 0.0) tp = NormalizeDouble(tp, _Digits);

   // price=0.0 -> CTrade fills at the current market price.
   bool ok = (type == ORDER_TYPE_BUY) ? g_trade.Buy(lot, _Symbol, 0.0, sl, tp)
                                      : g_trade.Sell(lot, _Symbol, 0.0, sl, tp);
   uint rc = g_trade.ResultRetcode();
   if(!ok || (rc != TRADE_RETCODE_DONE && rc != TRADE_RETCODE_DONE_PARTIAL))
      Print((type == ORDER_TYPE_BUY ? "Buy" : "Sell"), " ", lot, " ", _Symbol,
            " failed: retcode=", rc, " (", g_trade.ResultRetcodeDescription(), ")");
}

//+------------------------------------------------------------------+
//| Main loop: act once per newly closed bar.                        |
//+------------------------------------------------------------------+
void OnTick()
{
   // ----- Act once per newly closed bar -----
   datetime curBarTime = iTime(_Symbol, _Period, 0);
   if(curBarTime == 0) return;                  // series not ready yet
   static datetime lastBarTime = 0;
   if(curBarTime == lastBarTime) return;

   // ----- Indicator must have produced data -----
   if(BarsCalculated(g_indHandle) <= 0) return; // still warming up; retry next tick

   // ----- Read entry signals from the last CLOSED bar (index 1) -----
   bool goLong = false, goShort = false;
   if(!ReadSignal(BUF_BUY,  goLong))  return;   // data not ready -> retry (don't consume the bar)
   if(!ReadSignal(BUF_SELL, goShort)) return;

   // The bar is now confirmed ready; mark it processed.
   lastBarTime = curBarTime;

   // ----- Read exit signals (exitLong = close longs; exitShort = close shorts) -----
   bool exitLong = false, exitShort = false;
   if(InpUseIndicatorExits)
   {
      ReadSignal(BUF_EXIT_BUY,  exitLong);      // best-effort: missing data => no exit this bar
      ReadSignal(BUF_EXIT_SELL, exitShort);
   }

   if(!goLong && !goShort && !exitLong && !exitShort) return; // nothing to do

   // ----- Trading must be permitted -----
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))     return;
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))           return;
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return;
   if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))   return;

   // ----- Current position state -----
   bool haveBuy = false, haveSell = false;
   CountPositions(haveBuy, haveSell);

   // ----- Indicator-driven exits first (close before any reversal) -----
   if(exitLong  && haveBuy)  { if(ClosePositions(POSITION_TYPE_BUY))  haveBuy  = false; }
   if(exitShort && haveSell) { if(ClosePositions(POSITION_TYPE_SELL)) haveSell = false; }

   // ----- Entries / reversals (Buy and Sell arrows are mutually exclusive) -----
   if(goLong)
   {
      if(haveSell && ClosePositions(POSITION_TYPE_SELL)) haveSell = false;
      if(!haveBuy && !haveSell) OpenPosition(ORDER_TYPE_BUY);
   }
   else if(goShort)
   {
      if(haveBuy && ClosePositions(POSITION_TYPE_BUY)) haveBuy = false;
      if(!haveSell && !haveBuy) OpenPosition(ORDER_TYPE_SELL);
   }
}

//+------------------------------------------------------------------+
//| Custom optimization criterion: trade-gated Sortino ratio          |
//| Runs once at the end of each pass. Select "Custom max" as the      |
//| optimization criterion to use it. MT5 has no built-in Sortino, so  |
//| we rebuild the per-trade return series and compute it here.        |
//+------------------------------------------------------------------+
double OnTester()
{
   // 1) Reject thin samples. Sortino on a handful of trades is noise.
   //    This is what kills the lucky low-trade passes.
   if(TesterStatistics(STAT_TRADES) < InpMinTrades)
      return 0.0;

   // 2) Rebuild the per-trade realized-return series from deal history.
   if(!HistorySelect(0, TimeCurrent()))
      return 0.0;

   int    total = HistoryDealsTotal();
   double rets[];
   ArrayResize(rets, total);
   int    n   = 0;
   double sum = 0.0;

   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      // Only deals that REALIZE profit/loss: close, reversal, or close-by.
      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT && entry != DEAL_ENTRY_OUT_BY)
         continue;

      double pl = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                + HistoryDealGetDouble(ticket, DEAL_SWAP)
                + HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      rets[n++] = pl;
      sum += pl;
   }

   if(n < InpMinTrades)
      return 0.0;

   double mean = sum / n;

   // 3) Target downside deviation (MAR = 0: any losing trade is "downside").
   //    Denominator uses N (all trades), the standard Sortino convention.
   double downsideSq = 0.0;
   for(int i = 0; i < n; i++)
   {
      double d = rets[i];           // d - target, with target = 0
      if(d < 0.0) downsideSq += d * d;
   }
   double downsideDev = MathSqrt(downsideSq / n);

   // 4) No losing trades at all (rare past the gate): Sortino is +inf,
   //    so cap to a large finite score to keep the optimizer stable.
   if(downsideDev <= 0.0)
      return (mean > 0.0) ? 1.0e6 : 0.0;

   return mean / downsideDev;        // the Sortino ratio
}
//+------------------------------------------------------------------+
