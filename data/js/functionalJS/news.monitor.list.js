Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.Monitor = Netdania.News.Monitor || {};

Netdania.News.Monitor.Notification = Netdania.News.Monitor.Notification || {};

Netdania.News.Monitor.Notification.Render.List = (function () {

    var _config = (function () {
        var _notification_config = Netdania.News.Monitor.Notification.ConfigNotification;
        var _list_config = {
            maxDisplayEventsList: 10,
            maxRequestEventsList: 10,
            timeoutVerificationSeconds: 5,
            readStatusSeconds: 5,
            signInLinkList: '<div class="signin">Please <a href="/UI/UserLogin.aspx">sign in</a> to view notifications</div>'
        }
        return $.extend(_notification_config, _list_config);
    })();

    /*Try to get the data form the data stored in object*/
    var _searchCache = function (data) {
        var control = data.control;
        var end = data.end;
        var headlines = data.headlines;
        var foundData = [];
        $.each(control.source, function (s) {
            var _disabled = control.source[s].userDisabled || false;
            foundData = foundData.concat(($.grep(control.news.data, function (v) {
                return (v.t <= end && v.source == control.source[s].source && v.provider == control.source[s].provider && !_disabled);
            })));
        });

        if (foundData.length >= headlines) {
            _updateNews(control);
            return true;
        }
        return false;
    };

    /*
     Generate the options information
     */
    var _optionsInfo = function (control, init) {
        try {

            if (init){
                /*
                 Generate mark all read button events
                 */
                var _readAllButton = $('#' + control.list.allReadButtonID);
                var _readAllButtonText = '<span>[Mark all as read]</span>';
                _readAllButton.html(_readAllButtonText);
                _readAllButton.bind("click", {
                    image: control
                }, function (event) {
                    var image = event.data.image;

                    /*
                     We mark every news as read
                     */
                    $.each(image.news.data, function (i) {
                        image.news.data[i].read = true;
                    });

                    _optionsInfo(image, false);

                    _eventNotificationCount(image);
                });
            }

            var panelVisible = false;

            /*
             Check for unread items
             */
            var unreadItems = $.grep(control.news.data, function (a) {
                return (!a.disabled && a.read == false);
            });

            if (unreadItems.length > 0) {
                $("#" +  control.list.allReadButtonID).show();
                panelVisible = true;
            } else {
                $("#" +  control.list.allReadButtonID).hide();
            }

            if (panelVisible) {
                $("#" +  control.list.optionsPanelID).show();
            } else {
                $("#" +  control.list.optionsPanelID).hide();
            }
        }
        catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.List.optionsInfo", err);
        }
    }

    /*
     Set the event for the More button
     */
    var _listMoreInfo = function (control) {
        try {

            var list = control.list;
            var _moreButton = $('#' + list.moreButtonID);
            var _moreButtonText = '<div class="button-container"><div id="spinner" class="spinner"></div><span>More</span></div>';
            _moreButton.html(_moreButtonText);
            _moreButton.bind("click", {
                control: control
            }, function (event) {
                try {
                    if (!event.data.control.list.moreClicked) {
                        $('#spinner').show();
                        event.data.control.list.moreClicked = true;
                        event.data.control.list.maxDisplayEvents += event.data.control.list.eventsIncrement;
                        //0 at start means that we do not care about the first news

                        var data = {control: event.data.control, start: 0, end: event.data.control.list.lastNewsTime - 1, headlines: event.data.control.list.maxRequestEvents };
                        if (!_searchCache(data)) {
                            event.data.control.searchCallback(data);
                        }
                    }
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.List.MoreClick", err);
                }
            });

            /*
             Header click for return to top
             */
            var _header = $('#' + list.headerPanelID);
            _header.bind("click", {
                control: control
            }, function (event) {
                try {
                    Netdania.Util.scrollToElement('#' + event.data.control.list.contentContainerPanelID, '#' + 'notification_table_' + event.data.control.id);
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.List.HeaderClick", err);
                }
            });

        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.List._listMoreInfo", err);
        }
    }

    /*
    Render the control container
     */
    var _displayBoxList = function (control) {

        var list = control.list;

        var _containerBox = list.container || "";
        var _title = list.title || "";
        var _footer = list.footer || "";
        var _content = list.html || "";

        var _contentPanelID = list.contentPanelID || "";
        var _footerPanelID = list.footerPanelID || "";
        var _countPanelID = list.countPanelID || "";
        var _moreButtonID = list.moreButtonID || "";
        var _contentContainerPanelID = list.contentContainerPanelID || "";
        var _headerID = list.headerPanelID || "";
        var _htmlContent;

        _htmlContent =
            '<div class="nd-notification-list-container">' +
                '<div class="nd-notification-list-header" id="' + _headerID + '">{Title}<div id="' + _countPanelID + '" class="nd-news-list-count"></div></div>' +
                '<div class="nd-notification-box-options" id="' + list.optionsPanelID + '"><div class="nd-notification-box-header-read" id="' + list.allReadButtonID + '"></div></div>' +
                '<div class="nd-notification-list-content-container" id="' + _contentContainerPanelID + '">' +
                '<div class="nd-notification-list-content" id="' + _contentPanelID + '">{Content}</div>' +
                '<div class="nd-notification-list-more" id="' + _moreButtonID + '"></div>' +
                '</div>' +
                '<div class="nd-notification-list-footer" id="' + _footerPanelID + '">{Footer}</div>' +
        '</div>';

        _htmlContent = _htmlContent.replace('{Title}', _title);
        _htmlContent = _htmlContent.replace('{Content}', _content);
        _htmlContent = _htmlContent.replace('{Footer}', _footer);

        /*
         If we have a container we add cod to the container
         */
        if (_containerBox != "") {
            $("#" + _containerBox).html(_htmlContent);
        }
    }

    /*
     Display the news box on user click
     */
    var _displayNewsList = function (control) {

        try {
            control.list.title = "Notifications";

            //set the id of the content div
            control.list.contentPanelID = 'content-' + control.list.id;
            control.list.headerPanelID = 'header-' + control.list.id;
            control.list.footerPanelID = 'footer-' + control.list.id;
            control.list.countPanelID = 'count-' + control.list.id;
            control.list.moreButtonID = 'more-' + control.list.id;
            control.list.contentContainerPanelID = 'content-container-' + control.list.id;

            control.list.optionsPanelID = 'options-panle-' + control.list.id;
            control.list.allReadButtonID = 'read-button-' + control.list.id;


            //Add news html to the container
            _displayBoxList(control);

            /*
             Display the more button
             */
            _listMoreInfo(control);


            _optionsInfo(control, true);

        }
        catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.List.DisplayNewsList", err);
        }

    }

    /*
     Display the news list on the detail page
     */
    var _updateNews = function (control) {
        try {

            if (control.user.authenticated) {
                //Generate the news html
                Netdania.News.Monitor.News.getInstance().WriteNewsList(control);

                //save the news marked as read in the storage
                Netdania.News.Monitor.Storage.UpdateNews(control);

                $('#' + control.list.footerPanelID).html("Showing " + control.list.displayedNews + ' notifications');
            }
            else {
                control.list.htmlText = _config.signInLinkList;
            }

            if (control.list.displayedNews == 0 || (control.list.limit != null && control.list.limit != undefined && control.list.limit)) {
                $('#' + control.list.moreButtonID).hide();
            } else {
                    $('#' + control.list.moreButtonID).show();
            }



            $('#' + control.list.contentPanelID).html(control.list.htmlText);

            if (control.list.moreClicked) {
                $('#spinner').hide();
                Netdania.Util.scrollToElement('#' + control.list.contentContainerPanelID, '#' + control.list.moreButtonID);
                control.list.moreClicked = false;
            }
        }
        catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.NewsList._updateNews", err);
        }

    }

    /*
     Update the notification of unread items
     */
    var _notificationUpdate = function (control) {
        try {
            if (control.news != null && control.news != undefined && control.news.data != undefined) {

                //Update the number of unread items
                var unreadItems = $.grep(control.news.data, function (a) {
                    return (!a.disabled && a.read == false);
                });
                if (unreadItems.length > 0) {
                    $("#" + control.list.countPanelID).show();
                    $("#" + control.list.countPanelID).html(unreadItems.length);
                }
                else {
                    $("#" + control.list.countPanelID).hide();
                }
            }
        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.Notification.List.notificationUpdate", err);
        }
    }

    /*
     Set news read after 10 seconds
     */
    var _timeOutNotification = function (control) {
        setTimeout(
            function () {

                /*
                 Call update event for all controls
                 */
                if (control.list.readNews != null && control.list.readNews != undefined && control.list.readNews > 0) {
                    _eventNotificationCount(control);
                }

                //update the notification count
                _notificationUpdate(control);


                //update the news read rows
                $("#" + control.list.container + " .unread").each(function () {
                    $(this).removeClass("unread");
                });

            }, _config.timeoutVerificationSeconds * 1000);
    }

    var _dataUpdateList = function (control) {
        try {
            _notificationUpdate(control);
            _timeOutNotification(control);
            _updateNews(control);
            _optionsInfo(control, false);
        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.Notification.List.dataUpdateList", err);
        }
    }


    var _eventNotificationCount = function (control) {
        try {
            control.NotificationCountEventCall(control);
        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.Notification.List.updateControls", err);
        }
    }

    var _init = function (data) {
        try {
            if (data != undefined) {

                var _controls = _config.notificationImages;
                var _news = Netdania.News.Monitor.News.getInstance();

                var imageID = data.image;
                var containerID = data.container;
                var maxDisplayEvents = data.maxDisplayEvents || _config.maxDisplayEventsList;
                var maxRequestEvents = data.maxRequestEvents || _config.maxRequestEventsList;
                var control;
                /*
                 If the control is not specified we get the first image from the list
                 */
                if (imageID == undefined) {
                    if (_controls.length > 0) {
                        control = _controls[0];
                    }
                    else {
                        alert('No notification is specified. Control is not configured.');
                        Netdania.Util.LogError("No notification image is specified", '');
                    }
                }
                else {
                    $.each(_controls, function (i) {
                        if (_controls[i].id == imageID) {
                            control = _controls[i].i;
                            return false;
                        }
                    })
                }

                if (control != undefined) {
                    if (control.enabled) {
                    if (containerID != undefined) {

                        var list = {
                            id: 1,
                            maxRequestEvents: maxRequestEvents,
                            container: containerID,
                            updateCallback: this.NewsDataUpdate,
                            maxDisplayEvents: maxDisplayEvents,
                            eventsIncrement: maxDisplayEvents,
                            displayedNews: 0
                        };

                        control.control.addListener(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_UPDATE, function (control) {
                           if (control!=null & control != undefined){
                               _dataUpdateList(control);
                           }
                        });

                        control.control.addListener(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_SOURCES_UPDATE, function (control) {
                            if (control!=null & control != undefined){
                                _updateNews(control);
                            }
                        });

                        control.control.addListener(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_COUNT_UPDATE, function (control) {
                            if (control != null & control != undefined) {
                                _dataUpdateList(control);
                            }
                        });

                        //add list to the control
                        control.list = list;

                        //call control news request
                        control.headlineCallback({control:control, headlines:list.maxDisplayEvents});

                        _displayNewsList(control);

                        _dataUpdateList(control);
                    }
                    else {
                        alert('No container is specified. Control is not configured.');
                        Netdania.Util.LogError("No container is specified", '');
                    }
                    }
                } else {
                    alert('Notification control is not set. Control is not configured.');
                    Netdania.Util.LogError("Image is not set", '');
                }
            }
            else {
                alert('No data is specified. Control is not configured.');
                Netdania.Util.LogError("No data is specified", '');
            }

        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.Notification.List", err);
        }
    }

    return _init;
})();














