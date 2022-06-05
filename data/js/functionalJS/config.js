Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.General = Netdania.News.General || {};
Netdania.News.General.Config = function () {

    this.NewsStory = "/Products/News/NewsStory.aspx";
    this.FullQuote = "/Products/FullQuote/FullQuote.aspx";
    this.IMAGES_PATH = "/JSComponents/news/General/images/";

    this.NEWS_DEFAULT = this.IMAGES_PATH + "sources/general_news.png";

    this.NEWS_SOURCES = [
    //RO
    //rssnews
        {"p": "rssnews", "s": "JurnalulBani", "imgURL": this.IMAGES_PATH + "sources/JurnalulBani.jpg", "alias": "rssnews JurnalulBani" },
    //US
    //FxWire
        {"p": "fxwire_dl", "s": "fxWire", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-EM", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-GN", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-CUR", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-CM", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-SI", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-GP", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-EI", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-RNA", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-MM", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-CB", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-EC", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-BN", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-TC", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-CF", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-BF", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire", "s": "fxWire@NC-EF", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro RT", "linkURL": "http://www.fxwirepro.com/" },



        { "p": "fxwire_dl", "s": "fxWire", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-EM", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-GN", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-CUR", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-CM", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-SI", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-GP", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-EI", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-RNA", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-MM", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-CB", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-EC", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-BN", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-TC", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-CF", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-BF", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },
        { "p": "fxwire_dl", "s": "fxWire@NC-EF", "imgURL": this.IMAGES_PATH + "sources/FxWire_Pro.gif", "alias": "FxWire Pro D:15", "linkURL": "http://www.fxwirepro.com/" },


    //rssnews
        {"p": "rssnews", "s": "ForbesMarkets", "imgURL": this.IMAGES_PATH + "sources/ForbesWorld.jpg", "alias": "Forbes Markets" },
        { "p": "rssnews", "s": "ForbesWorld", "imgURL": this.IMAGES_PATH + "sources/ForbesWorld.jpg", "alias": "Forbes World" },
        { "p": "rssnews", "s": "ForbesBonds", "imgURL": this.IMAGES_PATH + "sources/ForbesWorld.jpg", "alias": "Forbes Bonds" },
        { "p": "rssnews", "s": "ForbesPersonalFinance", "imgURL": this.IMAGES_PATH + "sources/ForbesWorld.jpg", "alias": "Forbes Personal Finance" },
        { "p": "rssnews", "s": "ForbesOptions", "imgURL": this.IMAGES_PATH + "sources/ForbesWorld.jpg", "alias": "Forbes Options" },

        //NASDAQ
        { "p": "rssnews", "s": "NASDAQStocks", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "NASDAQ Stocks" },
        { "p": "rssnews", "s": "NASDAQForex", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "NASDAQ Forex" },
        { "p": "rssnews", "s": "NASDAQUSMarkets", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "NASDAQ US Markets" },
        { "p": "rssnews", "s": "NASDAQEconomy", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "NASDAQ Economy" },
        { "p": "rssnews", "s": "NASDAQInternational", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "NASDAQ International" },
        { "p": "rssnews", "s": "NASDAQCommodities", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "NASDAQ Commodities" },

        { "p": "rssnews", "s": "BBCUS_Canada", "imgURL": this.IMAGES_PATH + "sources/general_news.png", "alias": "BBC US-Canada" },

    //UK
    //mni
        {"p": "mni", "s": "FX_Basic", "imgURL": this.IMAGES_PATH + "sources/mni.png", "alias": "MNI" },
        {"p": "mni", "s": "FX_Basic_Alerts", "imgURL": this.IMAGES_PATH + "sources/mni.png", "alias": "MNI" },
    //rssnews
        {"p": "rssnews", "s": "BBCBusiness", "imgURL": this.IMAGES_PATH + "sources/BBC.jpg", "alias": "BBCBusiness" },
        { "p": "rssnews", "s": "BBCPolitics", "imgURL": this.IMAGES_PATH + "sources/BBC.jpg", "alias": "BBCPolitics" },
        { "p": "rssnews", "s": "Marketwatch", "imgURL": this.IMAGES_PATH + "sources/MarketWatch.jpg", "alias": "MarketWatch" },
    //DK
    //aktiefokus
        {"p": "aktiefokus", "s": "AktiefokusMain", "imgURL": this.IMAGES_PATH + "sources/AktiefokusMain.jpg", "alias": "Aktiefokus RT" },
        {"p": "aktiefokus", "s": "AktiefokusMain_Alerts", "imgURL": "sources/AktiefokusMain.jpg", "alias": "Aktiefokus" },
        { "p": "aktiefokus", "s": "AktiefokusDelayed", "imgURL": this.IMAGES_PATH + "sources/AktiefokusMain.jpg", "alias": "Aktiefokus D:15" },
        { "p": "rb-borsen", "s": "RBB_StockExchange", "imgURL": this.IMAGES_PATH + "sources/AktiefokusMain.jpg", "alias": "RBB StockExchange RT" },
        { "p": "aktiefokus_calendar", "s": "", "imgURL": this.IMAGES_PATH + "sources/AktiefokusMain.jpg", "alias": "Aktiefokus Calendar" },
        { "p": "signals", "s": "SignalServer", "imgURL": this.IMAGES_PATH + "sources/netdania.png", "alias": "Netdania Notify", "openFQ": true },

        { "p": "netdania_newschannel", "s": "Channel1_news_test",  "imgURL": this.IMAGES_PATH + "sources/netdania.png", "alias": "Netdania Test" }
    ];

    this.NEWS_BANKS = [
    //RBS, Societe Generale, RBC, Barclays, Standard Chartered 
        {"bank": "RBS", "imgURL": this.IMAGES_PATH + "banks/RBS.jpg", "alias": "Royal Bank of Scotland" },
        { "bank": "Societe Generale", "imgURL": this.IMAGES_PATH + "banks/SocieteGenerale.jpg", "alias": "Societe Generale" },
        { "bank": "RBC", "imgURL": this.IMAGES_PATH + "banks/RBC.jpg", "alias": "Royal Bank of Canada" },
        { "bank": "Barclays", "imgURL": this.IMAGES_PATH + "banks/Barclays.jpg", "alias": "Barclays" },
        { "bank": "Standard Chartered", "imgURL": this.IMAGES_PATH + "banks/StandardChartered.jpg", "alias": "Standard Chartered" },
        { "bank": "HSBC Global Research", "imgURL": this.IMAGES_PATH + "banks/hsbc.jpeg", "alias": "HSBC Global Research" }
    ];

    this.CURRENCY_PATH = "categories/currency/";
    this.NEWS_IMAGES_0 = [
        {"size": "0", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "0/currency_1.png" },
        {"size": "0", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "0/currency_2.png" },
        {"size": "0", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "0/currency_3.png" },
        {"size": "0", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "0/currency_4.png" }
    ];
    this.NEWS_IMAGES_1 = [
        {"size": "1", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "1/currency_1.png" },
        {"size": "1", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "1/currency_2.png" },
        {"size": "1", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "1/currency_3.png" },
        {"size": "1", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "1/currency_4.png" }
    ];
    this.NEWS_IMAGES_2 = [
        {"size": "2", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "2/currency_1.png" },
        {"size": "2", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "2/currency_2.png" },
        {"size": "2", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "2/currency_3.png" },
        {"size": "2", "type": "currency", "imgURL": this.IMAGES_PATH + this.CURRENCY_PATH + "2/currency_4.png" }
    ];
};

