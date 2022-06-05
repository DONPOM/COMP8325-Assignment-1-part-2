Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.Utilities = Netdania.News.Utilities || {};


Netdania.News.Utilities.getRandomizer = function(bottom, top) {
        return Math.floor( Math.random() * ( 1 + top - bottom ) ) + bottom;
}

Netdania.News.Utilities.getHeadlineImage = function (source, relatedInstrumentsJSON) {

    var config = new Netdania.News.General.Config();
    
    if (relatedInstrumentsJSON !== undefined && relatedInstrumentsJSON.Inf !== undefined && relatedInstrumentsJSON.Inf.RNA !== "") {
        
        var newsBanks = config.NEWS_BANKS;
        for (var i = 0; i < newsBanks.length; i++) {
            if (newsBanks[i].bank == relatedInstrumentsJSON.Inf.RNA) {
                return newsBanks[i].imgURL;
            }
        }
    }

    var newsSources = config.NEWS_SOURCES;
    for (var i = 0; i < newsSources.length; i++) {
        if (newsSources[i].s == source) {
            return newsSources[i].imgURL;
        }
    }

    return config.NEWS_DEFAULT;
}

Netdania.News.Utilities.getLinkBySource = function (source, relatedInstrumentsJSON) {
    var config = new Netdania.News.General.Config();

    if (relatedInstrumentsJSON !== undefined && relatedInstrumentsJSON.Inf !== undefined && relatedInstrumentsJSON.Inf.RNA !== "") {
        var newsBanks = config.NEWS_BANKS;
        for (var i = 0; i < newsBanks.length; i++) {
            if (newsBanks[i].bank == relatedInstrumentsJSON.Inf.RNA && newsBanks[i].linkURL !== undefined) {
                return newsBanks[i].linkURL;
            }
        }
        return '';
    }
    else {
        var newsSources = config.NEWS_SOURCES;
        for (var i = 0; i < newsSources.length; i++) {
            if (newsSources[i].s == source && newsSources[i].linkURL !== undefined) {
                return newsSources[i].linkURL;
            }
        }
    }
    return '';
}

Netdania.News.Utilities.getAliasBySource = function (source, relatedInstrumentsJSON) {
    var config=new Netdania.News.General.Config();

    if (relatedInstrumentsJSON !== undefined && relatedInstrumentsJSON.Inf !== undefined && relatedInstrumentsJSON.Inf.RNA !== "") {
        var newsBanks = config.NEWS_BANKS;
        for (var i = 0; i < newsBanks.length; i++) {
            if (newsBanks[i].bank == relatedInstrumentsJSON.Inf.RNA) {
                return newsBanks[i].alias;
            }
        }
    }

    var newsSources=config.NEWS_SOURCES;
    for(var i=0;i<newsSources.length;i++)
    {
        if(source.indexOf(newsSources[i].s) != -1)
        {
            return newsSources[i].alias;
        }
    }
    return '';
}


Netdania.News.Utilities.getAlias = function (data) {
    var source = data.source;
    var relatedInstrumentsJSON = data.instrument;
    var provider = data.provider || "";

    var config=new Netdania.News.General.Config();

    if (relatedInstrumentsJSON !== undefined && relatedInstrumentsJSON.Inf !== undefined && relatedInstrumentsJSON.Inf.RNA !== "") {
        var newsBanks = config.NEWS_BANKS;
        for (var i = 0; i < newsBanks.length; i++) {
            if (newsBanks[i].bank == relatedInstrumentsJSON.Inf.RNA) {
                return newsBanks[i].alias;
            }
        }
    }

    var newsSources=config.NEWS_SOURCES;
    for(var i=0;i<newsSources.length;i++)
    {
        if(source.indexOf(newsSources[i].s) != -1 && (provider != "" && provider.indexOf(newsSources[i].p) != -1))
        {
            return newsSources[i].alias;
        }
    }
    return '';
}


Netdania.News.Utilities.getOpenFQSource = function (source) {
    var config = new Netdania.News.General.Config();

    var newsSources = config.NEWS_SOURCES;
    for (var i = 0; i < newsSources.length; i++) {
        if (newsSources[i].s == source && newsSources[i].openFQ !== undefined) {
            return newsSources[i].openFQ;
        }
    }
    return '';
}

Netdania.News.Utilities.getRandomNewsImage = function (imageSize, imageType) {

    var config = new Netdania.News.General.Config();

    var newsImages = config.NEWS_IMAGES_0;

    switch(imageSize)
    {
        case 0:
            newsImages = config.NEWS_IMAGES_0;
            break;
        case 1:
            newsImages = config.NEWS_IMAGES_1;
            break;
        default:
            newsImages = config.NEWS_IMAGES_2;
    }


    var i = Netdania.News.Utilities.getRandomizer(0,newsImages.length-1);
    return newsImages[i].imgURL;

    return config.NEWS_DEFAULT;
}

Netdania.News.Utilities.timeDifference = function (current, previous) {
    var msPerMinute = 60 * 1000;
    var msPerHour = msPerMinute * 60;
    var msPerDay = msPerHour * 24;
    var msPerMonth = msPerDay * 30;
    var msPerYear = msPerDay * 365;

    var elapsed = current - previous;

    if (elapsed < msPerMinute) {
        var seconds = Math.round(elapsed/1000);
        if (seconds < 0){
            seconds = 0;
        }
        return  seconds + ' seconds ago';
    }

    else if (elapsed < msPerHour) {
        return Math.round(elapsed/msPerMinute) + ' minutes ago';
    }

    else if (elapsed < msPerDay ) {
        return Math.round(elapsed/msPerHour ) + ' hours ago';
    }

    else if (elapsed < msPerMonth) {
        return 'approximately ' + Math.round(elapsed/msPerDay) + ' days ago';
    }

    else if (elapsed < msPerYear) {
        return 'approximately ' + Math.round(elapsed/msPerMonth) + ' months ago';
    }

    else {
        return 'approximately ' + Math.round(elapsed/msPerYear ) + ' years ago';
    }
}
