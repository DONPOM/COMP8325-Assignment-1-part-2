var Netdania = Netdania || {};
Netdania.General = Netdania.General || {};
Netdania.General.Config = Netdania.General.Config || {};
Netdania.General.Config.Scheme = ('https:' == document.location.protocol ? 'https://' : 'http://');
Netdania.General.Config.Host =  Netdania.General.Config.Scheme + "balancer11.netdania.com/StreamingServer/StreamingServer";
//Netdania.General.Config.Host = Netdania.General.Config.Scheme +  "internet.netdania.com/StreamingServer/StreamingServer";
Netdania.General.Config.FullQuoteURL = "/Products/FullQuote/FullQuote.aspx";
Netdania.General.Config.MobileJSURL = "/Products/JavascriptChart/MobileJavascriptChart.aspx";
Netdania.General.Config.FinanceChartURL = "/Products/FinanceChart/FinanceChart.aspx";
Netdania.General.Config.JavascriptChartURL = "/Products/JavascriptChart/JavascriptChart.aspx";
Netdania.General.Config.QuoteListURL = "/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx";
Netdania.General.Config.NewsURL = "/Products/News/ResearchAnalysis.aspx";
Netdania.General.Config.NewsTechnicalURL = "/Products/News/Technicals.aspx";
Netdania.General.Config.NewsCalendarURL = "/Products/News/Calendar.aspx";
Netdania.General.Config.DenmarkStocksURL = "/Products/QuoteList/Stocks/Denmark.aspx";
Netdania.General.Config.InstrumentNews = "/Products/News/InstrumentNews.aspx";

Netdania.General.Config.URLConnectionParams = [
    { url: 'Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx', ct: Netdania.JsApi.ConnectionType.POLLING, pollingInterval: 1000 }
    , { url: 'Products/QuoteList/QuoteListFullscreen.aspx', ct: Netdania.JsApi.ConnectionType.LONGPOLLING, pollingInterval: 1000 }
    //,{ url: '', ct: Netdania.JsApi.ConnectionType.POLLING }
];

Netdania.General.Config.GetConnectionParams = function (url, connectionTypes) {
    
    var newUrl = url;

    var indexOfBackSlash = url.indexOf('/');
    if (indexOfBackSlash === 0) {
        newUrl = url.substring(1, url.length);
    }

    for (var i = 0; i < connectionTypes.length; i++) {
        if (connectionTypes[i].url === newUrl) {
            return connectionTypes[i];
            break;
        }
    }
    return { url: '', ct: Netdania.JsApi.ConnectionType.POLLING, pollingInterval: 2000 }
};

