//+------------------------------------------------------------------+
//| SignalAlerts.mqh                                                 |
//| Confirmed entry-signal alert formatting and delivery.            |
//+------------------------------------------------------------------+
#ifndef LORENTZIAN_CLASSIFICATION_SIGNAL_ALERTS_MQH
#define LORENTZIAN_CLASSIFICATION_SIGNAL_ALERTS_MQH

enum ENUM_LC_SIGNAL_DIRECTION
{
   LC_SIGNAL_SELL = -1,
   LC_SIGNAL_BUY  = 1
};

struct SignalAlertState
{
   datetime lastBuyBarTime;
   datetime lastSellBarTime;
};

//+------------------------------------------------------------------+
//| Reset per-instance deduplication state.                          |
//+------------------------------------------------------------------+
void ResetSignalAlertState(SignalAlertState &state)
{
   state.lastBuyBarTime  = 0;
   state.lastSellBarTime = 0;
}

//+------------------------------------------------------------------+
//| Return a compact chart-timeframe label, for example H1 or D1.    |
//+------------------------------------------------------------------+
string SignalAlertTimeframe(const ENUM_TIMEFRAMES timeframe)
{
   ENUM_TIMEFRAMES resolved = timeframe == PERIOD_CURRENT
                              ? (ENUM_TIMEFRAMES)Period()
                              : timeframe;
   return StringSubstr(EnumToString(resolved), 7);
}

//+------------------------------------------------------------------+
//| Format the user-facing alert message.                            |
//+------------------------------------------------------------------+
string BuildSignalAlertMessage(const ENUM_LC_SIGNAL_DIRECTION direction,
                               const string symbol,
                               const ENUM_TIMEFRAMES timeframe,
                               const datetime signalBarTime,
                               const double closePrice,
                               const int digits)
{
   string directionText = direction == LC_SIGNAL_BUY ? "BUY" : "SELL";
   return StringFormat("LDC %s | %s | %s | %s | Close: %s",
                       directionText,
                       symbol,
                       SignalAlertTimeframe(timeframe),
                       TimeToString(signalBarTime, TIME_DATE | TIME_MINUTES),
                       DoubleToString(closePrice, digits));
}

//+------------------------------------------------------------------+
//| Claim a live signal event once per direction and bar timestamp.  |
//+------------------------------------------------------------------+
bool ClaimSignalAlert(SignalAlertState &state,
                      const ENUM_LC_SIGNAL_DIRECTION direction,
                      const datetime signalBarTime,
                      const bool eligible)
{
   if(!eligible || signalBarTime <= 0)
      return false;

   if(direction == LC_SIGNAL_BUY)
   {
      if(state.lastBuyBarTime == signalBarTime)
         return false;
      state.lastBuyBarTime = signalBarTime;
      return true;
   }

   if(direction == LC_SIGNAL_SELL)
   {
      if(state.lastSellBarTime == signalBarTime)
         return false;
      state.lastSellBarTime = signalBarTime;
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Deliver one confirmed signal alert and optional mobile push.     |
//+------------------------------------------------------------------+
bool DispatchConfirmedSignalAlert(SignalAlertState &state,
                                  const ENUM_LC_SIGNAL_DIRECTION direction,
                                  const datetime signalBarTime,
                                  const double closePrice,
                                  const bool enableSignalAlerts,
                                  const bool enablePushNotifications,
                                  const bool eligible)
{
   if(!enableSignalAlerts ||
      !ClaimSignalAlert(state, direction, signalBarTime, eligible))
      return false;

   string message = BuildSignalAlertMessage(direction, _Symbol, _Period,
                                            signalBarTime, closePrice, _Digits);
   Print("LDC signal event: ", message);

   // MetaTrader documents all external notification functions as unavailable
   // in Strategy Tester. Keep the event visible in its Journal without
   // implying that a terminal alert or push was delivered there.
   if((bool)MQLInfoInteger(MQL_TESTER))
   {
      Print("LDC: terminal and push alerts are not delivered in Strategy Tester.");
      return true;
   }

   Alert(message);

   if(enablePushNotifications)
   {
      if(!(bool)TerminalInfoInteger(TERMINAL_NOTIFICATIONS_ENABLED))
      {
         Print("LDC: push notification skipped because terminal notifications are not configured or enabled.");
      }
      else
      {
         ResetLastError();
         if(!SendNotification(message))
            PrintFormat("LDC: SendNotification failed with error %d.",
                        GetLastError());
      }
   }

   return true;
}

#endif // LORENTZIAN_CLASSIFICATION_SIGNAL_ALERTS_MQH
