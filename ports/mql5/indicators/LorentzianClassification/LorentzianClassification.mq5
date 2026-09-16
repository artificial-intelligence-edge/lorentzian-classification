//+------------------------------------------------------------------+
//|                                    LorentzianClassification.mq5  |
//| Machine Learning: Lorentzian Classification                       |
//+------------------------------------------------------------------+
//
// ====================
// ==== Background ====
// ====================
//
// When classifying market state via neighbor selection, choosing an
// appropriate distance metric is essential. Euclidean Distance is often
// used as the default distance metric, but it may not always be the best
// choice. This is because market data is often significantly impacted by
// proximity to significant world events such as FOMC Meetings and Black
// Swan events. These major economic events can contribute to a warping
// effect analogous a massive object's gravitational warping of Space-Time.
// In financial markets, this warping effect operates on a continuum, which
// can analogously be referred to as "Price-Time".
//
// To help to better account for this warping effect, Lorentzian Distance
// can be used as an alternative distance metric to Euclidean Distance. The
// geometry of Lorentzian Space can be difficult to visualize at first, and
// one of the best ways to intuitively understand it is through an example
// involving 2 feature dimensions (z=2). For purposes of this example, let's
// assume these two features are Relative Strength Index (RSI) and the
// Average Directional Index (ADX). In reality, the optimal number of
// features is in the range of 3-8, but for the sake of simplicity, we will
// use only 2 features in this example.
//
// Fundamental Assumptions:
// (1) We can calculate RSI and ADX for a given chart.
// (2) For simplicity, values for RSI and ADX are assumed to adhere to a
//     Gaussian distribution in the range of 0 to 100.
// (3) The most recent RSI and ADX value can be considered the origin of a
//     coordinate system with ADX on the x-axis and RSI on the y-axis.
//
// Distances in Euclidean Space:
// Measuring the Euclidean Distances of historical values with the most
// recent point at the origin will yield a distribution in which the nearest
// neighbors cluster spherically around the origin.
//
// Distances in Lorentzian Space:
// However, the same set of historical values measured using Lorentzian
// Distance will yield a different distribution in which the neighborhood
// is warped outward along both axes.
//
// Observations:
// (1) In Lorentzian Space, the shortest distance between two points is not
//     necessarily a straight line, but rather, a geodesic curve.
// (2) The warping effect of Lorentzian distance reduces the overall
//     influence of outliers and noise.
// (3) Lorentzian Distance becomes increasingly different from Euclidean
//     Distance as the number of nearest neighbors used for comparison
//     increases.
//
// =================================
// ==== Next Bar Classification ====
// =================================
//
// This model specializes specifically in predicting the direction of price
// action over the course of the next 4 bars. To avoid complications with
// the ML model, this value is hardcoded to 4 bars but support for other
// training lengths may be added in the future.
//
#property copyright   "AI Edge"
#property link        "https://ai-edge.io/"
#property version     "1.01"
#property description "ML-based indicator using Lorentzian distance with greedy ANN neighbor selection."
#property indicator_chart_window
#property indicator_buffers 13
#property indicator_plots   8

// Plot 0: Buy Signal (drawn via OBJ_ARROW_BUY objects; buffer kept for Data Window)
#property indicator_label1  "Buy Signal"
#property indicator_type1   DRAW_NONE

// Plot 1: Sell Signal (drawn via OBJ_ARROW_SELL objects; buffer kept for Data Window)
#property indicator_label2  "Sell Signal"
#property indicator_type2   DRAW_NONE

// Plot 2: Exit Buy
#property indicator_label3  "Exit Buy"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  C'0,153,136'
#property indicator_width3  2

// Plot 3: Exit Sell
#property indicator_label4  "Exit Sell"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  C'204,51,17'
#property indicator_width4  2

// Plot 4: Kernel Estimate (per-bar colored line: bullish slope green, bearish red)
#property indicator_label5  "Kernel Estimate"
#property indicator_type5   DRAW_COLOR_LINE
#property indicator_color5  C'0,153,136',C'204,51,17'
#property indicator_width5  2

// Plots 5-6: hidden data buffers
#property indicator_label6  "Direction"
#property indicator_type6   DRAW_NONE
#property indicator_label7  "Prediction"
#property indicator_type7   DRAW_NONE

// Plot 7: Color Candles (uses 5 buffers: OHLC + color)
#property indicator_label8   "Bar Colors"
#property indicator_type8    DRAW_COLOR_CANDLES
#property indicator_style8   STYLE_SOLID
#property indicator_width8   1

// Includes
#include "Include/MLFeatures.mqh"
#include "Include/KernelFunctions.mqh"
#include "Include/Filters.mqh"
#include "Include/ANN.mqh"
#include "Include/Backtest.mqh"
#include "Include/SignalAlerts.mqh"

// =====================================================================
// ================
// ==== Inputs ====
// ================
//
// General Settings: Settings object for user-defined general inputs.
// Feature Engineering: User-defined inputs for the feature series used as
// the input to the ML model. The default feature set provides a reasonable
// starting point, but the optimal set typically varies by instrument and
// timeframe.
// Filters: User-defined filters for adjusting the frequency and shape of
// the ML model's predictions.
// Kernel Settings: Parameters for the Nadaraya-Watson kernel regression
// used as an optional prediction filter and visual overlay.
// Display Settings: Cosmetic options for the chart.
//
// Trade Stats note: The trade stats section is NOT intended to be used as
// a replacement for proper backtesting. It is intended to be used for
// calibration purposes only.
// =====================================================================
// ===================== General Settings =====================
// NOTE: 'input group' is intentionally NOT used. In this MT5 build an
// 'input group' line consumes an iCustom() positional parameter slot,
// which silently shifts every following input when the indicator is
// driven by an EA via iCustom. Keep these as plain comments.
input ENUM_APPLIED_PRICE InpSource = PRICE_CLOSE; // Source: Source of the input data
input int    InpNeighborsCount   = 8;     // Neighbors Count: Number of neighbors to consider (1-100)
input int    InpMaxBarsBack      = 2000;  // Max Bars Back: Max historical bars used for ML lookups
input int    InpFeatureCount     = 5;     // Feature Count: Features used for ML predictions (2-5)
input int    InpColorCompression = 1;     // Color Compression: color intensity factor (1-10)
input bool   InpShowExits        = false; // Show Default Exits: fixed exit 4 bars after an entry signal
input bool   InpUseDynamicExits  = false; // Use Dynamic Exits: adjust exit threshold via kernel regression
input bool   InpIncludeFullHist  = false; // Include Full History: train ANN on all bars, not just recent
input bool   InpShowTradeStats   = true;  // Show Trade Stats: table for calibration only, NOT a backtest
input bool   InpUseWorstCase     = false; // Use Worst Case Estimates: close-only estimates in trade stats

// ===================== Filters =====================
// Prediction Filters: Used for adjusting the frequency of the ML model's
// predictions. The default ON set is Volatility + Regime. ADX, EMA, and
// SMA are OFF by default and useful for additional trend confluence.
input bool   InpUseVolFilter     = true;  // Use Volatility Filter: recentATR(1) > historicalATR(10)
input bool   InpUseRegimeFilter  = true;  // Use Regime Filter: Kaufman Adaptive Slope, trending vs ranging
input bool   InpUseAdxFilter     = false; // Use ADX Filter: require ADX > threshold for signal flips
input double InpRegimeThreshold  = -0.1;  // Regime Threshold: (-10 to 10); higher = require stronger trend
input int    InpAdxThreshold     = 20;    // ADX Threshold: (0-100); higher = require stronger trend
input bool   InpUseEmaFilter     = false; // Use EMA Filter: long only when close > EMA, short when below
input int    InpEmaPeriod        = 200;   // EMA Period: Period of the EMA used for the EMA filter
input bool   InpUseSmaFilter     = false; // Use SMA Filter: long only when close > SMA, short when below
input int    InpSmaPeriod        = 200;   // SMA Period: Period of the SMA used for the SMA filter

// ===================== Kernel Settings =====================
// Nadaraya-Watson Kernel Regression Settings
input bool   InpUseKernelFilter    = true;  // Trade with Kernel: gate entries by kernel direction
input bool   InpShowKernelEst      = true;  // Show Kernel Estimate: plot kernel regression on the chart
input bool   InpUseKernelSmoothing = false; // Enhance Kernel Smoothing: fewer color flips, more ML entries
input int    InpKernelH            = 8;     // Lookback Window: (3-50); sliding window of recent bars
input double InpKernelR            = 8.0;   // Relative Weighting: ->0 favors long timeframes; ->inf Gaussian
input int    InpKernelX            = 25;    // Regression Level: (2-25); smaller=tighter, larger=looser fit
input int    InpKernelLag          = 2;     // Lag: crossover detection (1-2); lower = earlier crossovers

// ===================== Feature Engineering =====================
// Feature Series: The ML model takes up to 5 features as inputs. The
// default feature set (RSI, WT, CCI, ADX, RSI-short) is a well-tested
// default configuration. ParamA is the primary parameter (usually the
// main lookback); ParamB is the secondary parameter (a smoothing period
// for RSI/CCI/WT, or unused for ADX).
input ENUM_FEATURE_TYPE InpF1Type = FEATURE_RSI; // Feature 1: The first feature to use for ML predictions
input int    InpF1ParamA = 14;   // Parameter A: Primary parameter of feature 1
input int    InpF1ParamB = 1;    // Parameter B: Secondary parameter of feature 1 (if applicable)
input ENUM_FEATURE_TYPE InpF2Type = FEATURE_WT;  // Feature 2: The second feature to use for ML predictions
input int    InpF2ParamA = 10;   // Parameter A: Primary parameter of feature 2
input int    InpF2ParamB = 11;   // Parameter B: Secondary parameter of feature 2 (if applicable)
input ENUM_FEATURE_TYPE InpF3Type = FEATURE_CCI; // Feature 3: The third feature to use for ML predictions
input int    InpF3ParamA = 20;   // Parameter A: Primary parameter of feature 3
input int    InpF3ParamB = 1;    // Parameter B: Secondary parameter of feature 3 (if applicable)
input ENUM_FEATURE_TYPE InpF4Type = FEATURE_ADX; // Feature 4: The fourth feature to use for ML predictions
input int    InpF4ParamA = 20;   // Parameter A: Primary parameter of feature 4
input int    InpF4ParamB = 2;    // Parameter B: Secondary parameter of feature 4 (if applicable)
input ENUM_FEATURE_TYPE InpF5Type = FEATURE_RSI; // Feature 5: The fifth feature to use for ML predictions
input int    InpF5ParamA = 9;    // Parameter A: Primary parameter of feature 5
input int    InpF5ParamB = 1;    // Parameter B: Secondary parameter of feature 5 (if applicable)

// ===================== Display Settings =====================
input bool   InpShowBarColors     = true;  // Show Bar Colors: Color each bar by the ML model's prediction
input bool   InpShowBarPreds      = true;  // Show Bar Prediction Values: integer prediction on each bar
input bool   InpUseAtrOffset      = false; // Use ATR Offset: ATR offset instead of prediction offset
input double InpBarPredOffset     = 0;     // Bar Prediction Offset: % offset from the bar high/low

// ===================== Alert Settings =====================
// Appended after the original 45 inputs so existing positional iCustom()
// callers continue to receive the same parameter mapping.
input bool   InpEnableSignalAlerts       = false; // Enable Signal Alerts: terminal alert on a confirmed Buy/Sell signal
input bool   InpEnablePushNotifications  = false; // Enable Push Notifications: also send to configured MetaQuotes IDs

// =====================================================================
// Indicator Buffers
// =====================================================================
double BuyBuf[], SellBuf[], ExitBuyBuf[], ExitSellBuf[];
double KernelBuf[], KernelColor[];
double DirectionBuf[], PredictionBuf[];
double CandleO[], CandleH[], CandleL[], CandleC[], CandleColor[];

// =====================================================================
// Global State
// =====================================================================
FeatureWork g_feat1, g_feat2, g_feat3, g_feat4, g_feat5;
FilterState g_filters;
ANNState    g_ann;
BacktestState g_backtest;
SignalAlertState g_signalAlerts;

// Resolved source price array (based on InpSource)
double g_src[];
// Working arrays for derived price types
double g_hlc3[];
double g_ohlc4[];
// ATR(1) for display offsets
double g_atr1TR[], g_atr1[];

// Kernel yhat arrays (computed per bar in loop)
double g_yhat1[], g_yhat2[];

// Color palette indices
// 0-9: green shades (9=strongest), 10-19: red shades (19=strongest), 20: neutral
#define CLR_GREEN_BASE   0
#define CLR_RED_BASE     10
#define CLR_NEUTRAL      20
#define CLR_PALETTE_SIZE 21

//+------------------------------------------------------------------+
//| Gradient color index for positive (green) predictions.            |
//| Mirrors a linear gradient from neutral toward teal (#009988)      |
//| with saturation controlled by neighborsCount/colorCompression.    |
//+------------------------------------------------------------------+
int GetGreenColorIdx(double pred)
{
   double compressionFactor = (double)InpNeighborsCount / (double)InpColorCompression;
   if(compressionFactor <= 0) compressionFactor = 1.0;
   double ratio = MathAbs(pred) / compressionFactor;
   int scaled = (int)MathMin(MathRound(ratio * 9.0), 9.0);
   return CLR_GREEN_BASE + MathMax(scaled, 0);
}

int GetRedColorIdx(double pred)
{
   double compressionFactor = (double)InpNeighborsCount / (double)InpColorCompression;
   if(compressionFactor <= 0) compressionFactor = 1.0;
   double ratio = MathAbs(pred) / compressionFactor;
   int scaled = (int)MathMin(MathRound(ratio * 9.0), 9.0);
   return CLR_RED_BASE + MathMax(scaled, 0);
}

//+------------------------------------------------------------------+
//| Custom indicator initialization                                  |
//+------------------------------------------------------------------+
int OnInit()
{
   // Map buffers to plots (order must match #property definitions).
   // DRAW_COLOR_LINE for the kernel plot consumes 2 buffers (data + color
   // index) at slots 4 + 5; subsequent buffers shift down by one.
   SetIndexBuffer(0,  BuyBuf,        INDICATOR_DATA);
   SetIndexBuffer(1,  SellBuf,       INDICATOR_DATA);
   SetIndexBuffer(2,  ExitBuyBuf,    INDICATOR_DATA);
   SetIndexBuffer(3,  ExitSellBuf,   INDICATOR_DATA);
   SetIndexBuffer(4,  KernelBuf,     INDICATOR_DATA);
   SetIndexBuffer(5,  KernelColor,   INDICATOR_COLOR_INDEX);
   SetIndexBuffer(6,  DirectionBuf,  INDICATOR_DATA);
   SetIndexBuffer(7,  PredictionBuf, INDICATOR_DATA);
   SetIndexBuffer(8,  CandleO,       INDICATOR_DATA);
   SetIndexBuffer(9,  CandleH,       INDICATOR_DATA);
   SetIndexBuffer(10, CandleL,       INDICATOR_DATA);
   SetIndexBuffer(11, CandleC,       INDICATOR_DATA);
   SetIndexBuffer(12, CandleColor,   INDICATOR_COLOR_INDEX);

   // Arrow codes
   PlotIndexSetInteger(0, PLOT_ARROW, 233); // up arrow
   PlotIndexSetInteger(1, PLOT_ARROW, 234); // down arrow
   PlotIndexSetInteger(2, PLOT_ARROW, 251); // x-cross
   PlotIndexSetInteger(3, PLOT_ARROW, 251);

   // Empty values (plots 0-6)
   for(int p = 0; p < 7; p++)
      PlotIndexSetDouble(p, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   // Color candle palette: 10 green shades + 10 red shades + neutral
   PlotIndexSetInteger(7, PLOT_COLOR_INDEXES, CLR_PALETTE_SIZE);
   // Green shades (teal #009988 = C'0,153,136') - lighten toward white
   for(int g = 0; g < 10; g++)
   {
      double frac = (g + 1) / 10.0; // 0.1 to 1.0
      int r  = (int)(255 - frac * 255);
      int gn = (int)(255 - frac * (255 - 153));
      int b  = (int)(255 - frac * (255 - 136));
      PlotIndexSetInteger(7, PLOT_LINE_COLOR, CLR_GREEN_BASE + g,
                          (color)((b << 16) | (gn << 8) | r));
   }
   // Red shades (#CC3311 = C'204,51,17')
   for(int rd = 0; rd < 10; rd++)
   {
      double frac = (rd + 1) / 10.0;
      int r  = (int)(255 - frac * (255 - 204));
      int gn = (int)(255 - frac * (255 - 51));
      int b  = (int)(255 - frac * (255 - 17));
      PlotIndexSetInteger(7, PLOT_LINE_COLOR, CLR_RED_BASE + rd,
                          (color)((b << 16) | (gn << 8) | r));
   }
   // Neutral gray
   PlotIndexSetInteger(7, PLOT_LINE_COLOR, CLR_NEUTRAL, C'120,123,134');

   // Initialize feature engines
   InitFeatureWork(g_feat1, InpF1Type, InpF1ParamA, InpF1ParamB, MathPow(10, _Digits));
   InitFeatureWork(g_feat2, InpF2Type, InpF2ParamA, InpF2ParamB, MathPow(10, _Digits));
   InitFeatureWork(g_feat3, InpF3Type, InpF3ParamA, InpF3ParamB, MathPow(10, _Digits));
   InitFeatureWork(g_feat4, InpF4Type, InpF4ParamA, InpF4ParamB, MathPow(10, _Digits));
   InitFeatureWork(g_feat5, InpF5Type, InpF5ParamA, InpF5ParamB, MathPow(10, _Digits));

   InitANN(g_ann);
   InitBacktest(g_backtest);
   ResetSignalAlertState(g_signalAlerts);

   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Cleanup                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "LC_Pred_");
   ObjectsDeleteAll(0, "LC_Buy_");
   ObjectsDeleteAll(0, "LC_Sell_");
   DeleteStatsTable();
}

//+------------------------------------------------------------------+
//| Main calculation                                                 |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   // Need at least 5 bars for training labels
   if(rates_total < 10) return 0;

   // Forward indexing
   ArraySetAsSeries(open, false);   ArraySetAsSeries(high, false);
   ArraySetAsSeries(low, false);    ArraySetAsSeries(close, false);
   ArraySetAsSeries(time, false);

   int begin = (prev_calculated > 0) ? prev_calculated - 1 : 0;

   // On full recalculation, reset all state
   if(prev_calculated == 0)
   {
      InitANN(g_ann);
      InitBacktest(g_backtest);
      ResetSignalAlertState(g_signalAlerts);
      InitFeatureWork(g_feat1, InpF1Type, InpF1ParamA, InpF1ParamB, MathPow(10, _Digits));
      InitFeatureWork(g_feat2, InpF2Type, InpF2ParamA, InpF2ParamB, MathPow(10, _Digits));
      InitFeatureWork(g_feat3, InpF3Type, InpF3ParamA, InpF3ParamB, MathPow(10, _Digits));
      InitFeatureWork(g_feat4, InpF4Type, InpF4ParamA, InpF4ParamB, MathPow(10, _Digits));
      InitFeatureWork(g_feat5, InpF5Type, InpF5ParamA, InpF5ParamB, MathPow(10, _Digits));
      ObjectsDeleteAll(0, "LC_Pred_");
      ObjectsDeleteAll(0, "LC_Buy_");
      ObjectsDeleteAll(0, "LC_Sell_");
   }

   // Resize working arrays
   ArrayResize(g_src, rates_total);
   ArrayResize(g_hlc3, rates_total);
   ArrayResize(g_ohlc4, rates_total);
   ArrayResize(g_atr1TR, rates_total);     ArrayResize(g_atr1, rates_total);
   ArrayResize(g_yhat1, rates_total);      ArrayResize(g_yhat2, rates_total);

   ResizeFeatureWork(g_feat1, rates_total);
   ResizeFeatureWork(g_feat2, rates_total);
   ResizeFeatureWork(g_feat3, rates_total);
   ResizeFeatureWork(g_feat4, rates_total);
   ResizeFeatureWork(g_feat5, rates_total);
   ResizeFilterState(g_filters, rates_total);

   // Resolve source price array based on InpSource
   for(int i = begin; i < rates_total; i++)
   {
      g_hlc3[i]  = (high[i] + low[i] + close[i]) / 3.0;
      g_ohlc4[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;
      switch(InpSource)
      {
         case PRICE_OPEN:      g_src[i] = open[i];    break;
         case PRICE_HIGH:      g_src[i] = high[i];    break;
         case PRICE_LOW:       g_src[i] = low[i];     break;
         case PRICE_MEDIAN:    g_src[i] = (high[i] + low[i]) / 2.0; break;
         case PRICE_TYPICAL:   g_src[i] = g_hlc3[i];  break;
         case PRICE_WEIGHTED:  g_src[i] = (high[i] + low[i] + close[i] + close[i]) / 4.0; break;
         default:              g_src[i] = close[i];    break; // PRICE_CLOSE
      }
   }

   // Compute features (RSI/CCI use g_src, WT uses g_hlc3, ADX uses high/low/g_src)
   CalcFeature(g_feat1, g_src, high, low, g_hlc3, begin, rates_total);
   CalcFeature(g_feat2, g_src, high, low, g_hlc3, begin, rates_total);
   CalcFeature(g_feat3, g_src, high, low, g_hlc3, begin, rates_total);
   CalcFeature(g_feat4, g_src, high, low, g_hlc3, begin, rates_total);
   CalcFeature(g_feat5, g_src, high, low, g_hlc3, begin, rates_total);

   // Compute filters
   CalcVolatilityFilter(high, low, close, g_filters, begin, rates_total);
   CalcRegimeFilter(g_ohlc4, high, low, g_filters, begin, rates_total);
   CalcADXFilter(high, low, close, 14, g_filters, begin, rates_total);
   CalcEMAFilter(close, InpEmaPeriod, g_filters, begin, rates_total);
   CalcSMAFilter(close, InpSmaPeriod, g_filters, begin, rates_total);

   // ATR(1) used for bar-prediction label offsets when InpUseAtrOffset is on.
   CalcATR(high, low, close, 1, g_atr1, g_atr1TR, begin, rates_total);

   // Signal state (persistent across bars via static)
   static int    signal     = 0;  // +1 long, -1 short, 0 neutral
   static int    barsHeld   = 0;
   static int    barsSinceStartLong  = 999999;
   static int    barsSinceStartShort = 999999;
   static int    barsSinceAlertBull  = 999999;
   static int    barsSinceAlertBear  = 999999;
   static double prevSignalChange  = 0;
   static double prevSignalChange1 = 0;
   static double prevSignalChange2 = 0;
   // Index of the last fully-processed CLOSED bar. The still-forming (last)
   // bar is deferred until it closes, so each bar is processed exactly once
   // with its final OHLC -- this keeps the incremental (chart/tester) result
   // identical to a full recalculation, prevents the ANN dataset from being
   // double-fed when a bar is reprocessed, and makes Buy/Sell signals persist
   // on closed bars so EA/iCustom readers can see them.
   static int    lastClosedBar = -1;

   if(prev_calculated == 0)
   {
      signal = 0; barsHeld = 0;
      barsSinceStartLong = 999999; barsSinceStartShort = 999999;
      barsSinceAlertBull = 999999; barsSinceAlertBear = 999999;
      prevSignalChange = 0; prevSignalChange1 = 0; prevSignalChange2 = 0;
      lastClosedBar = -1;
   }

   // ===== Main bar loop (CLOSED bars only) =====
   // Process each bar exactly once, when it has closed (final OHLC). The
   // still-forming last bar (index rates_total - 1) is deferred to the next
   // call; this is what makes signals persist on closed bars for readers.
   int mainStart = lastClosedBar + 1;
   if(mainStart < 0) mainStart = 0;
   for(int i = mainStart; i < rates_total - 1; i++)
   {
      // Default: empty all signal buffers
      BuyBuf[i] = LC_EMPTY;   SellBuf[i] = LC_EMPTY;
      ExitBuyBuf[i] = LC_EMPTY; ExitSellBuf[i] = LC_EMPTY;
      DirectionBuf[i] = 0;    PredictionBuf[i] = 0;

      // Color candles (copy OHLC, default neutral)
      if(InpShowBarColors)
      {
         CandleO[i] = open[i]; CandleH[i] = high[i];
         CandleL[i] = low[i];  CandleC[i] = close[i];
         CandleColor[i] = CLR_NEUTRAL;
      }
      else
      {
         CandleO[i] = LC_EMPTY; CandleH[i] = LC_EMPTY;
         CandleL[i] = LC_EMPTY; CandleC[i] = LC_EMPTY;
         CandleColor[i] = 0;
      }

      // =========================
      // ====  Core ML Logic  ====
      // =========================
      //
      // Training label for bar i: the direction of the 4-bar window ending
      // at bar i (backward-looking). The training horizon is hard-coded to
      // 4 to match the model's design; support for other lengths may be
      // added in the future.
      //
      // y_train = src[i-4] < src[i] ? short  (price rose      -> short label)
      //         : src[i-4] > src[i] ? long   (price declined  -> long  label)
      //         : neutral
      //
      // The mapping is contrarian: features similar to a historical period
      // that ended in a 4-bar rise cast a short vote for today, and vice
      // versa. The ANN loop then sums those votes into a net prediction.
      int trainLabel = 0;
      if(i >= 4)
         trainLabel = g_src[i - 4] < g_src[i] ? -1 : g_src[i - 4] > g_src[i] ? 1 : 0;

      // Push this bar's features and label into the ANN's rolling dataset.
      // Feature arrays are appended chronologically; their history becomes
      // the candidate pool for the nearest-neighbor search on the next bar.
      double fv1 = NZ(g_feat1.output[i]);
      double fv2 = NZ(g_feat2.output[i]);
      double fv3 = NZ(g_feat3.output[i]);
      double fv4 = NZ(g_feat4.output[i]);
      double fv5 = NZ(g_feat5.output[i]);
      ANNPushBar(g_ann, fv1, fv2, fv3, fv4, fv5, trainLabel);

      // Nadaraya-Watson Kernel Regression: yhat1 uses the Rational
      // Quadratic kernel; yhat2 is a Gaussian with lag applied for crossover
      // detection. The kernel values drive both the visual overlay and the
      // kernel-based prediction filter computed below.
      g_yhat1[i] = KernelRationalQuadratic(g_src, i, InpKernelH, InpKernelR, InpKernelX);
      int lagH = InpKernelH - InpKernelLag;
      if(lagH < 1) lagH = 1;
      g_yhat2[i] = KernelGaussian(g_src, i, lagH, InpKernelX);
      KernelBuf[i] = InpShowKernelEst ? g_yhat1[i] : LC_EMPTY;

      // Approximate Nearest Neighbors Search with Lorentzian Distance:
      // A novel variation of the Nearest Neighbors (NN) search algorithm
      // that ensures a chronologically uniform distribution of neighbors.
      // The full algorithm and rationale are documented in ANN.mqh. In
      // summary:
      //   1. The algorithm iterates through the dataset in chronological
      //      order, using the modulo operator to only perform calculations
      //      every 4 bars. This reduces computational overhead and ensures
      //      a minimum chronological spacing between neighbors of at least
      //      4 bars.
      //   2. A list of k-similar neighbors is simultaneously maintained in
      //      both a predictions array and a corresponding distances array.
      //   3. When the size of the predictions array exceeds the desired
      //      number of nearest neighbors (InpNeighborsCount), the algorithm
      //      removes the first neighbor from the predictions array and the
      //      corresponding distance array.
      //   4. The lastDistance variable is overridden to be a distance in
      //      the lower 25% of the array. This is intended to reduce the
      //      rate at which subsequent newly added distance values increase.
      //   5. Lorentzian distance is used in order to minimize the effect of
      //      outliers and take into account the warping of "price-time"
      //      due to proximity to significant economic events.
      // prediction = sum of neighbor labels (positive -> long, negative ->
      // short, magnitude = vote strength on a [-k, +k] scale).
      // The gate uses last_bar_index (the FINAL bar in the dataset), not the
      // running bar index, so RunANN fires only for the most recent maxBarsBack
      // bars. The greedy queue persists across calls, so starting earlier would
      // poison the neighbor pool with mis-aligned state.
      //   maxBarsBackIndex = last_bar_index >= maxBarsBack
      //                       ? last_bar_index - maxBarsBack : 0
      //   if bar_index >= maxBarsBackIndex: RunANN
      int maxBarsBackIdx = (rates_total - 1 >= InpMaxBarsBack)
                              ? (rates_total - 1) - InpMaxBarsBack : 0;
      double prediction = 0;
      if(i >= maxBarsBackIdx)
      {
         RunANN(g_ann, InpNeighborsCount, InpFeatureCount, InpMaxBarsBack,
                fv1, fv2, fv3, fv4, fv5, InpIncludeFullHist,
                rates_total - 1);
         prediction = g_ann.prediction;
      }
      PredictionBuf[i] = prediction;

      // ============================
      // ==== Prediction Filters ====
      // ============================
      //
      // User-Defined Filters: Used for adjusting the frequency of the ML
      // model's predictions. filterAll combines the three gating filters
      // (volatility, regime, ADX); only it can prevent a signal flip.
      // The EMA/SMA trend filters are applied later at the entry stage.
      bool filtVol     = GetVolatilityFilter(g_filters, i, InpUseVolFilter);
      bool filtRegime  = GetRegimeFilter(g_filters, i, InpRegimeThreshold, InpUseRegimeFilter);
      bool filtAdx     = GetADXFilter(g_filters, i, InpAdxThreshold, InpUseAdxFilter);
      bool filterAll   = filtVol && filtRegime && filtAdx;

      bool isEmaUp   = GetEMAUptrend(g_filters, close, i, InpUseEmaFilter);
      bool isEmaDn   = GetEMADowntrend(g_filters, close, i, InpUseEmaFilter);
      bool isSmaUp   = GetSMAUptrend(g_filters, close, i, InpUseSmaFilter);
      bool isSmaDn   = GetSMADowntrend(g_filters, close, i, InpUseSmaFilter);

      // Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel
      // Regression using the Rational Quadratic Kernel.
      //
      // Kernel Rates of Change compare yhat1[i-1] vs yhat1[i] (is-rate) and
      // yhat1[i-2] vs yhat1[i-1] (was-rate) to detect direction and flips.
      // Kernel Crossovers compare the smoothed yhat2 against the primary
      // yhat1 estimate, providing an alternative "smoother" trigger.
      bool isBullishRate = (i >= 2) && g_yhat1[i - 1] < g_yhat1[i];
      bool isBearishRate = (i >= 2) && g_yhat1[i - 1] > g_yhat1[i];
      bool wasBullishRate = (i >= 3) && g_yhat1[i - 2] < g_yhat1[i - 1];
      bool wasBearishRate = (i >= 3) && g_yhat1[i - 2] > g_yhat1[i - 1];
      bool isBullishChange = isBullishRate && wasBearishRate;
      bool isBearishChange = isBearishRate && wasBullishRate;
      bool isBullishCross  = (i >= 1) && g_yhat2[i] >= g_yhat1[i] && g_yhat2[i - 1] < g_yhat1[i - 1];
      bool isBearishCross  = (i >= 1) && g_yhat2[i] <= g_yhat1[i] && g_yhat2[i - 1] > g_yhat1[i - 1];
      bool isBullishSmooth = g_yhat2[i] >= g_yhat1[i];
      bool isBearishSmooth = g_yhat2[i] <= g_yhat1[i];

      // Alert Variables: which kernel event fires the bull/bear alert.
      // Bullish/Bearish Filters: the kernel gate applied to entries. With
      // the kernel filter OFF, everything is considered both bullish and
      // bearish (i.e. the kernel is not a gate).
      bool alertBullish = InpUseKernelSmoothing ? isBullishCross : isBullishChange;
      bool alertBearish = InpUseKernelSmoothing ? isBearishCross : isBearishChange;
      bool isBullish = InpUseKernelFilter ? (InpUseKernelSmoothing ? isBullishSmooth : isBullishRate) : true;
      bool isBearish = InpUseKernelFilter ? (InpUseKernelSmoothing ? isBearishSmooth : isBearishRate) : true;

      // Kernel line color: per-bar via DRAW_COLOR_LINE color buffer.
      //   index 0 = bullish (green), index 1 = bearish (red)
      //   colorByRate  = isBullishRate  ? c_green : c_red
      //   colorByCross = isBullishSmooth ? c_green : c_red
      bool kernelBullish = InpUseKernelSmoothing ? isBullishSmooth : isBullishRate;
      KernelColor[i] = kernelBullish ? 0 : 1;

      // Filtered Signal: the model's prediction of future price movement
      // direction with user-defined filters applied.
      //   signal := prediction > 0 and filter_all ? long
      //           : prediction < 0 and filter_all ? short
      //           : nz(signal[1])
      // Only volatility + regime + adx gate the signal flip; kernel/EMA/SMA
      // gates are applied later to the entry signal (startLongTrade).
      int prevSignal = signal;
      if(prediction > 0 && filterAll)       signal = 1;
      else if(prediction < 0 && filterAll)  signal = -1;
      // else signal holds previous value

      DirectionBuf[i] = signal;

      // Fractal Filters: derived from relative appearances of signals in a
      // given time-series fractal/segment with a default length of 4 bars.
      // An "early signal flip" is a change that occurred within the last
      // 3 bars of another change -- high counts can indicate choppy
      // (ranging) market conditions.
      double signalChange = (double)(signal - prevSignal);
      bool isDiffSignalType = signalChange != 0;
      bool isEarlyFlip = isDiffSignalType &&
                         (prevSignalChange != 0 || prevSignalChange1 != 0 || prevSignalChange2 != 0);

      // Shift signal change history (1-bar, 2-bar, 3-bar lags)
      prevSignalChange2 = prevSignalChange1;
      prevSignalChange1 = prevSignalChange;
      prevSignalChange  = signalChange;

      // Bar-Count Filters: represent strict filters based on a pre-defined
      // holding period of 4 bars. barsHeld resets to 0 on any signal flip
      // and increments otherwise.
      if(isDiffSignalType) barsHeld = 0;
      else                 barsHeld++;

      bool isHeldFourBars         = (barsHeld == 4);
      bool isHeldLessThanFourBars = (barsHeld > 0 && barsHeld < 4);

      // ===========================
      // ==== Entries and Exits ====
      // ===========================
      //
      // Entry Conditions: booleans for ML model position entries.
      //   isBuy        = signal == long AND isEmaUp AND isSmaUp
      //   startLong    = isBuy AND isDiffSignalType AND isBullish
      //                  (equivalently: signal flip in the bullish
      //                  direction that agrees with EMA/SMA/kernel gates)
      // At default settings (EMA and SMA filters disabled), isEmaUp/isSmaUp
      // are always true, so the kernel filter (isBullish) is the material
      // gate on top of the fractal filter (isDiffSignalType).
      bool isBuy  = (signal == 1)  && isEmaUp && isSmaUp;
      bool isSell = (signal == -1) && isEmaDn && isSmaDn;
      bool isNewBuySignal  = isBuy  && isDiffSignalType;
      bool isNewSellSignal = isSell && isDiffSignalType;
      bool startLong  = isNewBuySignal  && isBullish;
      bool startShort = isNewSellSignal && isBearish;

      bool isLastBuy  = (i >= 4) && (DirectionBuf[i - 4] == 1);
      bool isLastSell = (i >= 4) && (DirectionBuf[i - 4] == -1);

      // barsSince tracking: counters used by the dynamic exit logic.
      if(startLong)     barsSinceStartLong  = 0; else barsSinceStartLong++;
      if(startShort)    barsSinceStartShort = 0; else barsSinceStartShort++;
      if(alertBullish)  barsSinceAlertBull  = 0; else barsSinceAlertBull++;
      if(alertBearish)  barsSinceAlertBear  = 0; else barsSinceAlertBear++;

      // Fixed Exit Conditions: booleans for ML model position exits based
      // on bar-count filters. Default exits occur exactly 4 bars after an
      // entry signal, matching the model's 4-bar training horizon.
      bool endLongStrict  = ((isHeldFourBars && isLastBuy)  ||
                             (isHeldLessThanFourBars && isNewSellSignal && isLastBuy)) &&
                            (i >= 4 && BuyBuf[i - 4] != LC_EMPTY);
      bool endShortStrict = ((isHeldFourBars && isLastSell) ||
                             (isHeldLessThanFourBars && isNewBuySignal && isLastSell)) &&
                            (i >= 4 && SellBuf[i - 4] != LC_EMPTY);

      // Dynamic Exit Conditions: booleans for ML model position exits based
      // on fractal filters and kernel regression filters. These attempt to
      // let profits ride by dynamically adjusting the exit threshold based
      // on kernel regression logic.
      bool isValidLongExit  = barsSinceAlertBear > barsSinceStartLong;
      bool isValidShortExit = barsSinceAlertBull > barsSinceStartShort;
      bool endLongDynamic   = isBearishChange && (i > 0 && isValidLongExit);
      bool endShortDynamic  = isBullishChange && (i > 0 && isValidShortExit);

      // Select exit mode. Dynamic exits are only valid when EMA filter,
      // SMA filter and kernel smoothing are all disabled; otherwise fall
      // back to strict 4-bar exits to avoid conflicting trend constraints.
      bool isDynValid = !InpUseEmaFilter && !InpUseSmaFilter && !InpUseKernelSmoothing;
      bool endLong  = (InpUseDynamicExits && isDynValid) ? endLongDynamic  : endLongStrict;
      bool endShort = (InpUseDynamicExits && isDynValid) ? endShortDynamic : endShortStrict;

      // =========================
      // ==== Plotting Labels ====
      // =========================
      //
      // Fill the signal buffers that produce Buy/Sell/Exit arrows. These
      // do not repaint once the most recent bar has fully closed.
      double arrowOff = NZ(g_atr1[i]) * 1.2;
      if(startLong)
      {
         BuyBuf[i] = low[i];
         string bname = "LC_Buy_" + IntegerToString(i);
         ObjectCreate(0, bname, OBJ_TEXT, 0, time[i], low[i] - arrowOff);
         ObjectSetString(0, bname, OBJPROP_TEXT, "\x25B2");
         ObjectSetInteger(0, bname, OBJPROP_FONTSIZE, 16);
         ObjectSetString(0, bname, OBJPROP_FONT, "Arial");
         ObjectSetInteger(0, bname, OBJPROP_COLOR, C'0,153,136');
         ObjectSetInteger(0, bname, OBJPROP_ANCHOR, ANCHOR_UPPER);
         ObjectSetInteger(0, bname, OBJPROP_SELECTABLE, false);
         DispatchConfirmedSignalAlert(g_signalAlerts, LC_SIGNAL_BUY,
                                      time[i], close[i],
                                      InpEnableSignalAlerts,
                                      InpEnablePushNotifications,
                                      prev_calculated > 0 && i == rates_total - 2);
      }
      if(startShort)
      {
         SellBuf[i] = high[i];
         string sname = "LC_Sell_" + IntegerToString(i);
         ObjectCreate(0, sname, OBJ_TEXT, 0, time[i], high[i] + arrowOff);
         ObjectSetString(0, sname, OBJPROP_TEXT, "\x25BC");
         ObjectSetInteger(0, sname, OBJPROP_FONTSIZE, 16);
         ObjectSetString(0, sname, OBJPROP_FONT, "Arial");
         ObjectSetInteger(0, sname, OBJPROP_COLOR, C'204,51,17');
         ObjectSetInteger(0, sname, OBJPROP_ANCHOR, ANCHOR_LOWER);
         ObjectSetInteger(0, sname, OBJPROP_SELECTABLE, false);
         DispatchConfirmedSignalAlert(g_signalAlerts, LC_SIGNAL_SELL,
                                      time[i], close[i],
                                      InpEnableSignalAlerts,
                                      InpEnablePushNotifications,
                                      prev_calculated > 0 && i == rates_total - 2);
      }
      if(endLong  && InpShowExits)            ExitBuyBuf[i]  = high[i] + arrowOff;
      if(endShort && InpShowExits)            ExitSellBuf[i] = low[i]  - arrowOff;

      // =========================
      // ==== Display Signals ====
      // =========================
      //
      // Bar coloring: each bar is tinted by a gradient derived from the
      // current prediction. Positive predictions saturate toward teal
      // (#009988); negative predictions saturate toward red (#CC3311).
      // Weak predictions fade toward white. compressionFactor =
      // neighborsCount / colorCompression controls how quickly the
      // gradient saturates.
      if(InpShowBarColors)
      {
         if(prediction > 0)      CandleColor[i] = GetGreenColorIdx(prediction);
         else if(prediction < 0) CandleColor[i] = GetRedColorIdx(prediction);
         else                    CandleColor[i] = CLR_NEUTRAL;
      }

      // =====================
      // ==== Backtesting ====
      // =====================
      //
      // Real-time trade stats: a useful mechanism for obtaining real-time
      // feedback during Feature Engineering. This does NOT replace the
      // need to properly backtest.
      // Note: in this context a "Stop-Loss" is defined as an instance
      // where the ML Signal prematurely flips directions before an exit
      // signal can be generated -- these are tracked as Early Signal Flips.
      if(InpShowTradeStats && i >= maxBarsBackIdx)
         UpdateBacktest(g_backtest, high[i], low[i], open[i], close[i],
                        startLong, endLong, startShort, endShort,
                        isEarlyFlip, InpUseWorstCase);

   } // end main bar loop

   // ----- Defer the still-forming (last) bar -----
   // Its OHLC is not final until it closes, so it carries no confirmed signal
   // yet; it is computed on the next call once a newer bar exists. Keep its
   // buffers empty so nothing stale or provisional reaches readers (this also
   // removes any last-bar repainting).
   int formingBar = rates_total - 1;
   if(formingBar >= 0)
   {
      BuyBuf[formingBar]       = LC_EMPTY; SellBuf[formingBar]      = LC_EMPTY;
      ExitBuyBuf[formingBar]   = LC_EMPTY; ExitSellBuf[formingBar]  = LC_EMPTY;
      DirectionBuf[formingBar] = 0;        PredictionBuf[formingBar] = 0;
      KernelBuf[formingBar]    = LC_EMPTY; KernelColor[formingBar]  = 0;
      CandleO[formingBar] = LC_EMPTY; CandleH[formingBar] = LC_EMPTY;
      CandleL[formingBar] = LC_EMPTY; CandleC[formingBar] = LC_EMPTY;
      CandleColor[formingBar] = 0;
   }
   if(rates_total >= 2) lastClosedBar = rates_total - 2;

   // ===== Post-loop: objects drawn on the last bar only =====
   int last = rates_total - 1;

   // Bar Prediction Labels: show the ML model's evaluation of each bar as
   // an integer. Label position follows the ATR offset (if enabled) or a
   // percentage of (high+low)/2 controlled by InpBarPredOffset. Limited to
   // the most recent 500 bars to avoid clutter on long histories.
   if(InpShowBarPreds)
   {
      ObjectsDeleteAll(0, "LC_Pred_");
      int labelStart = MathMax(0, last - 500);
      for(int j = labelStart; j <= last; j++)
      {
         double pred = PredictionBuf[j];
         if(pred == 0) continue;
         string name = "LC_Pred_" + IntegerToString(j);
         double atrBase = NZ(g_atr1[j]);
         double totalOff;
         if(InpUseAtrOffset)
            totalOff = atrBase * 1.5;
         else
         {
            double pctOff = (high[j] + low[j]) / 2.0 * MathAbs(InpBarPredOffset) / 20.0;
            totalOff = atrBase * 1.2 + pctOff;
         }
         double yPos = pred > 0 ? high[j] + totalOff : low[j] - totalOff;
         ObjectCreate(0, name, OBJ_TEXT, 0, time[j], yPos);
         ObjectSetString(0, name, OBJPROP_TEXT, IntegerToString((int)pred));
         color pclr = pred > 0 ? C'0,153,136' : C'204,51,17';
         ObjectSetInteger(0, name, OBJPROP_COLOR, pclr);
         ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 10);
         ObjectSetString(0, name, OBJPROP_FONT, "Arial");
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, pred > 0 ? ANCHOR_LOWER : ANCHOR_UPPER);
      }
   }

   // Trade stats table
   if(InpShowTradeStats)
      DrawStatsTable(g_backtest);

   return rates_total;
}
//+------------------------------------------------------------------+
