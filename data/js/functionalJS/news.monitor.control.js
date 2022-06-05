Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.Monitor = Netdania.News.Monitor || {};


Netdania.News.Monitor.Notification = (function () {

    /*
     Local variable declaration
     */
    var _instance;

    var _config = {
        notificationImages: [],
        sources: [
            {
                source: 'FX_Basic_Alerts',
                provider: 'mni',
                alias: 'MNI',
                name: 'News Notifications',
                description: 'Top important news',
                login: false,
                useIfLoggedIn: true,
                countryFilter: false
            }
        ],
        maxRequestEvents: 5,
        maxDisplayEvents: 5,
        searchMaxNews:200,
        editSources: true,
        maxNewsDisplayDays: 30,
        displayOnSecured: false
    }


    var _render = {};

    /*
     Check for valid sources (require user to be logged in or specific country source)
     Get all news and disable the news with invalid sources
     */
    var _checkForValidSources = function(notificationImage) {

        try {
            $.each(notificationImage.source, function (i) {
                var _needsLogin = false;
                var _canSourceBeUsed = true;

                //check if the source needs the user be authenticated(login=true)
                _needsLogin = notificationImage.source[i].login || false;

                //for the sources that are used without login is possible to be removed when the user has logged in.
                var _useIfUserIsLoggedIn = notificationImage.source[i].useIfLoggedIn || false;


                _canSourceBeUsed = _needsLogin ? (notificationImage.user.authenticated ? true : false) : (notificationImage.user.authenticated ? _useIfUserIsLoggedIn : true);

                /*
                 If we have a country restriction parameter
                 We check if the user country match restriction and we remove the source otherwise
                 */
                var _countryRestriction = notificationImage.source[i].countryRestriction || false;
                if (_countryRestriction) {
                    var countryCode = notificationImage.user.countryCode;
                    if (_canSourceBeUsed && (countryCode == '' || countryCode.toLowerCase() != _countryRestriction.toLowerCase())) {
                        _canSourceBeUsed = false;
                    }
                }

                var _sourceDisabledByUser = notificationImage.source[i].userDisabled || false;

                notificationImage.source[i].isValid = _canSourceBeUsed

                if (notificationImage.news.data !== undefined) {
                $.each(notificationImage.news.data, function (s) {
                    if (notificationImage.source[i].source == notificationImage.news.data[s].source && notificationImage.source[i].provider == notificationImage.news.data[s].provider) {
                        if (!notificationImage.source[i].isValid || _sourceDisabledByUser) {
                        notificationImage.news.data[s].disabled = true;
                        }
                        else {
                            if (notificationImage.news.data[s].disabled && !_sourceDisabledByUser) {
                                notificationImage.news.data[s].disabled = false;
                            }
                        }
                    }
                });
                }
            });

        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor._checkForValidSources", err);
        }
    }

    /*
     Sync the data from control with the data from the storage
     */
    var _syncStorage = function (notificationImage) {
        try {
            /*
             Get data from storage
             */
            var data = {idControl:notificationImage.id, idUser:notificationImage.user.id, authenticated: notificationImage.user.authenticated};
            var storageNews = Netdania.News.Monitor.Storage.GetNewsData(data);

            notificationImage.news = notificationImage.news || {};

            if (storageNews != undefined && storageNews.source != undefined) {

                /*
                 Sync the news sources.
                 1. Check for user disabled sources
                 */
                $.each(storageNews.source, function (s) {
                    $.each(notificationImage.source, function (i) {
                        if (notificationImage.source[i].source == storageNews.source[s].source && notificationImage.source[i].provider == storageNews.source[s].provider) {
                            notificationImage.source[i].userDisabled = true;
                            return false;
                        }
                    });
                });

                /*
                 2. Get the latest news
                 */
                notificationImage.news.data = storageNews.data || [];


                /*
                 3. Get the latest news time. If is not set we det the current date
                 */
                notificationImage.news.lastNewsTime = storageNews.lastNewsTime || new Date().getTime() / 1000;


                /*
                 4. Get the first news time
                 */
                notificationImage.news.firstNewsTime = storageNews.firstNewsTime || new Date().getTime() / 1000;

            }


        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor._syncStorage", err);
        }
    };

    var _getImage = function(id)
    {

        var _image = $.grep(_config.notificationImages, function (a) {
            return a.id == id;
        })[0];

        return _image;
    }

    var _minTime = function(news)  {
        var lastNewsTime = new Date().getTime() / 1000;
        $.each(news, function (n) {

            if (lastNewsTime > news[n].t) {
                lastNewsTime = news[n].t
            }
        });
        return lastNewsTime;
    }

    var _getRequest = function(control, type) {
        control.request = control.request || {};
        control.request[type] = control.request[type] || {};
        return control.request[type];
    }

    /*
    Get all news from the request sort them and then keep ony the required number
     */
    var _filterData = function(request){
        var headlines = request.headlines;
        if (request.tempnews.news != null && request.tempnews.news != undefined) {

            //sort the new data array
            request.tempnews.news = request.tempnews.news.sort(function (a, b) {
                var _aT = a.t;
                var _bT = b.t;

                return _bT - _aT;
            });
            request.tempnews.news = request.tempnews.news.slice(0, headlines);
        }

        return request.tempnews;
    };

    /*
     Public functions declaration
     */
    return {
        Init: function (options) {

            _instance = this;
            Events.enable.call(this);

            var sources = options.s || _config.sources;
            var maxRequestEvents = options.maxRequestEvents || _config.maxRequestEvents;
            var maxDisplayEvents = options.maxDisplayEvents || _config.maxDisplayEvents;
            var containerID = options.container || "";
            var editSources = options.editSources || _config.editSources;
            var id = options.id || "";

            var images = _config.notificationImages;

            var _user = new Netdania.Util.User();

            var imageHtml = '';
            try {
                var idArrayCount = images.length;
                if (id == "") {
                    if (containerID == "") {
                        id = idArrayCount;
                    }
                    else {
                        id = Netdania.Util.HashCode(containerID);
                    }
                }
                var idContainerString = "news-notification-container-" + id;
                var idImageString = "news-notification-image-" + id;
                var idBoxString = "news-notification-box-" + id;
                var idCountString = "news-notification-count-" + id;


                images.push(
                    {
                        id: id,
                        source: sources,
                        imageId: idImageString,
                        containerId: idContainerString,
                        boxId: idBoxString,
                        countId: idCountString,
                        maxRequestEvents: maxRequestEvents,
                        maxDisplayEvents: maxDisplayEvents,
                        editSources: editSources,
                        updateCallback: this.NewsDataUpdate,
                        headlineCallback: this.NewsDataHeadlineRequest,
                        searchCallback: this.NewsDataSearchRequest,
                        updateEventCall: this.UpdateEventCall,
                        NotificationCountEventCall: this.NotificationCountEventCall,
                        NotificationSourcesEventCall: this.NotificationSourcesEventCall,
                        user: {
                                authenticated: _user.isAuthenticated(),
                                countryCode: _user.countryCode(),
                                id: _user.getUserId()
                        },
                        control: this,
                        enabled:(Netdania.Util.isSecure() ? _config.displayOnSecured : true),
                        alert: {},
                        maxNewsDisplayDays: _config.maxNewsDisplayDays
                    });

                var currentImage = images[idArrayCount];

                if (currentImage.enabled) {

                    imageHtml = '<div class="nd-news-notification-container" id="' + currentImage.containerId + '">' +
                        '<div onclick="Netdania.News.Monitor.Notification.Render.Popup.OnClick(this, event)" style="cursor:pointer;" class="nd-news-notification-image" id="' + currentImage.imageId + '" title="View notifications">' +
                        '<div class="nd-news-notification-count" id="' + currentImage.countId + '"></div>' +
                        '</div>' +
                        '<div class="nd-news-notification-box" id="' + currentImage.boxId + '"></div>' +
                        '</div>';

                    /*
                     Sync image with the data from storage
                     */
                    _syncStorage(currentImage);

                    /*
                    Check valid news sources
                    */
                    _checkForValidSources(currentImage);

                    /*
                     Make the news request to the server
                     */

                    if (currentImage.news.lastNewsTime != null && currentImage.news.lastNewsTime != undefined) {
                        /*
                         If we have already news we get the news from current time to the most recent news available at the client
                         */
                        var endDate = new Date().getTime() / 1000;
                        var startDate = currentImage.news.lastNewsTime + 1;
                        this.NewsDataSearchRequest({control: currentImage, headlines: _config.searchMaxNews, start: startDate, end: endDate});

                    }
                    /*
                     We make a headline request for monitoring
                     */
                    this.NewsDataHeadlineRequest({control: currentImage, headlines: currentImage.maxRequestEvents});

                    if (containerID != "") {
                        $("#" + containerID).html(imageHtml);
                        Netdania.News.Monitor.Notification.Render.Popup.Init(currentImage);
                    }
                    else {
                        return imageHtml;
                    }
                }
            }
            catch (err) {
                Netdania.Util.LogError("Netdania.News.Monitor.Init", err);
            }



        },
        NewsDataSearchRequest: function (data) {

            /*
             Get the request object
             */
            var request = _getRequest(data.control, Netdania.News.Monitor.News.Type.SEARCH);

            /*
             Remove old search requests
             */
            var _news = Netdania.News.Monitor.News.getInstance();
            _news.RemoveRequests(request);


            /*
             Initiate object
             */
            request.headlines = data.headlines;
            request.tempnews = {};
            request.data = Netdania.News.Monitor.News.getInstance().AddNewsSearchRequest(data);
            request.history = request.data.slice(0);

        },
        NewsDataHeadlineRequest: function (data) {
            var control = data.control;
            var _headlines = data.headlines;

            /*
            Get the headline request for the control
             */
            var request = _getRequest(control, Netdania.News.Monitor.News.Type.HEADLINE);

            //check for existing request
            if (request.headlines != null && request.headlines != undefined) {
                if (_headlines < request.headlines) {
                    _headlines = request.headlines;
                }
            }

            /*
            If there are headline (monitor) requests we remove them
             */
            var _news = Netdania.News.Monitor.News.getInstance();
            _news.RemoveRequests(request);

            //make the new request
            request.headlines = _headlines;
            request.tempnews = {};
            request.data = _news.AddNewsMonitorRequest(data);
            request.history = request.data.slice(0);

        },
        /*
        Receive news for all requests (multiple sources)
        When all requests receive data we send data to the views
         */
        NewsDataUpdate: function (newData) {
            try {
                if (newData.type == Netdania.News.Monitor.News.Type.HEADLINE || newData.type == Netdania.News.Monitor.News.Type.SEARCH) {
                    var image = _getImage(newData.control);
                    if (image != null && image != undefined) {
                        var dataReady = false;
                        var request = _getRequest(image,newData.type);
                        if (request.data != null && request.data != undefined && request.data.length>0) {
                            $.each(request.data, function (i) {
                                if (request.data[i] == newData.requestID) {

                                    //If the temporary news object is null add first data else append the news
                                    if ($.isEmptyObject(request.tempnews)) {
                                        request.tempnews = newData;
                                    } else {
                                        request.tempnews.news = request.tempnews.news.concat(newData.news);
                                    }

                                    //Remove the requested ID from the Que
                                    request.data.splice(i, 1);
                                    return false;
                                }
                            });

                            //if there are no requests left we can send the data
                            if (request.data.length == 0) {
                                dataReady = true;
                            }

                            if (dataReady) {
                                /*
                                 We sort and filter the request data the we send to process
                                 */
                                _instance.NewsDataReceived(_filterData(request));
                            }
                        }
                    }
                } else {

                    //on UPDATE we send the data directly
                    _instance.NewsDataReceived(newData);
                }
            }
            catch (err) {
                Netdania.Util.LogError("Netdania.News.Monitor.NewsDataUpdate", err);
            }
        },
        NewsDataReceived: function (data) {
            try {
                var news = data.news;

                /*
                Find the control
                 */
                var image =_getImage(data.control);

                $.each(news, function (j) {
                    try {


                            news[j]["read"] = false;
                            image.news = image.news || {};
                            image.news.data = image.news.data || [];

                            var found = false;
                            /*
                             Check if the news is already loaded and set the status
                             */

                            $.each(image.news.data, function (s) {
                                if (image.news.data[s].i == news[j].i) {
                                    /*
                                     Found a match.
                                     Set the news status
                                     */
                                    news[j]["read"] = image.news.data[s].read;
                                    found = true;
                                    return false;
                                }
                            });

                            /*
                            Check if the source is disabled(this is for alerts where we cannot remove the monitor request)
                             */
                          $.each(image.source, function (i) {
                              if (image.source[i].userDisabled && image.source[i].source == news[j].source && image.source[i].provider == news[j].provider) {
                                  news[j].disabled = true;
                              }
                            });

                        if (!found) {

                            /*
                             If the NEWS is SMALLER than the FIRST NEWS of the client, we mark the news as READ
                             */
                            if (image.news.firstNewsTime != undefined && news[j].t < image.news.firstNewsTime) {
                                news[j]["read"] = true;
                            }

                            var addNews = true;
                            /*If the user is not logged in we allow only a limited number of news*/
                            if (!image.user.authenticated && image.news.data.length >= image.maxDisplayEvents) {
                                addNews = false;
                            }

                            if (addNews) {
                                /*
                                 Add new data
                                 */
                                image.newData = true;
                                image.news.data.push(news[j]);
                            }
                        }

                    }
                    catch (err) {
                        Netdania.Util.LogError("Netdania.News.Monitor.NewsDataUpdate AddNews", err);
                    }
                });

                /*
                We sort the news
                 */
                if (image.newData && image.news != null && image.news != undefined && image.news.data != null && image.news.data != undefined) {

                    /*Maximum date to store (current date minus number of seconds)*/

                    var maxNewsDate = new Date().getTime() / 1000 - _config.maxNewsDisplayDays * 24 * 60 * 60;
                    var oldNewsCount = image.news.data.length;
                    image.news.data = $.grep(image.news.data, function( obj ) {
                        return obj.t >= maxNewsDate;
                    });

                    /*If data has been deleted (the limit has been reached) we signal the control*/
                    if (oldNewsCount > image.news.data.length){
                       image.news.limitID = image.news.data[image.news.data.length-1].i;
                    }

                    //sort the new data array
                    image.news.data = image.news.data.sort(function (a, b) {
                        var _aT = a.t;
                        var _bT = b.t;

                        return _bT - _aT;
                    });


                   /*
                   Calculate first and the last news time
                    */
                    var lastNewsTime = 0;
                    var firstNewsTime = new Date().getTime() / 1000;

                    $.each(image.news.data, function (n) {
                        if (lastNewsTime < image.news.data[n].t) {
                            lastNewsTime = image.news.data[n].t;
                        }

                        if (firstNewsTime > image.news.data[n].t) {
                            firstNewsTime = image.news.data[n].t;
                        }
                    });

                    image.news.lastNewsTime = lastNewsTime;
                    image.news.firstNewsTime = firstNewsTime;

                    Netdania.News.Monitor.Storage.UpdateNews(image);
                }

                _instance.UpdateEventCall(image);


            }
            catch (err) {
                Netdania.Util.LogError("Netdania.News.Monitor.NewsDataReceived", err);
            }
        },
        ConfigNotification: _config,
        Render: _render,
        Events: {
            NOTIFICATION_NEWS_UPDATE: 'OnNotificationNewsUpdate',
            NOTIFICATION_NEWS_COUNT_UPDATE: 'OnNotificationNewsCountUpdate',
            NOTIFICATION_NEWS_SOURCES_UPDATE: 'OnNotificationNewsSourcesUpdate'
        },
        UpdateEventCall : function(image){
            _instance.fireEvent(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_UPDATE, [image]);
        },
        NotificationCountEventCall : function(image){
            /*save to storage*/
            _instance.fireEvent(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_COUNT_UPDATE, [image]);
        },
        NotificationSourcesEventCall : function(image){
            _instance.fireEvent(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_SOURCES_UPDATE, [image]);
        },
        CheckForValidSources: _checkForValidSources

    }
})();














