
Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.Monitor = Netdania.News.Monitor || {};

Netdania.News.Monitor.Notification = Netdania.News.Monitor.Notification || {};

Netdania.News.Monitor.Notification.Render.Popup = (function () {

    var _config = (function() {
        var _notification_config = Netdania.News.Monitor.Notification.ConfigNotification;
        var _list_config = {
            Title: 'Notifications',
            Footer: '<a href="/Products/News/Notifications.aspx">View all notifications</a>',
            signInLinkDescriptionPopup: '<div class="signin">Please <a href="/UI/UserLogin.aspx">sign in</a> to view notifications</div>',
            signInLinkFooterPopup: '<a href="/UI/UserLogin.aspx">Sign in here</a>'
        }

        return $.extend(_notification_config, _list_config)
    })();

    /*
     Generate the debug information
     */
    var _debugInfo = function (Image) {

        if (queryString('debug00') != 'false') {
            var debugInfo = '<div id="debug' + Image.imageId + '"><h4>DEBUG INFORMATION</h4>';
            var html = '';
            /*
             The sources from configuration
             */
            if (Image.source != null && Image.source != undefined) {
                html = '<span><b>Config Sources</b> - set in the configuration file</span>' +
                    '<div>' +
                    '<p>' +
                    '<ul>';
                $.each(Image.source, function (i) {
                    html += '<li>' + Image.source[i].provider + '|' + Image.source[i].source + '</li>';
                });
                html += '</ul>' +
                    '</p>' +
                    '</div>';
                debugInfo += html;
            }
            /*
             The sources that are requested from server
             */

            html = '<span><b>Request Sources</b> - actual sources send to server (filtered with country, login request or user disabled)</span>' +
                '<div>' +
                '<p>' +
                '<ul>';
            $.each(Image.source, function (i) {
                if (Image.source[i].requested) {
                    html += '<li>' + Image.source[i].provider + '|' + Image.source[i].fullSource + '</li>';
                }
            });
            html += '</ul>' +
                '</p>' +
                '</div>';
            debugInfo += html;


            debugInfo += '</div>';


            /*
             Set the content to the corresponding debug panel id
             */
            $('#' + Image.news.debugPanelID).html(debugInfo);
        }
    }

    /*
     Generate the options information
     */
    var _optionsInfo = function (image, init) {
        try {

            if (init){
                /*
                Generate mark all read button events
                 */
                var _readAllButton = $('#' + image.news.allReadButtonID);
                var _readAllButtonText = '<span>[Mark all as read]</span>';
                _readAllButton.html(_readAllButtonText);
                _readAllButton.bind("click", {
                    image: image
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
            var unreadItems = $.grep(image.news.data, function (a) {
                return (!a.disabled && a.read == false);
            });

            if (unreadItems.length > 0) {
                $("#" +  image.news.allReadButtonID).show();
                panelVisible = true;
            } else {
                $("#" +  image.news.allReadButtonID).hide();
            }

            if (panelVisible) {
                $("#" +  image.news.optionsPanelID).show();
            } else {
                $("#" +  image.news.optionsPanelID).hide();
            }
        }
        catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.optionsInfo", err);
        }
    }

    /*
    Render the control container box
     */
    var _displayBoxPopup = function (notificationImage) {

        //set the id of the content div
        notificationImage.news.contentPanelID = 'content-' + notificationImage.imageId;

        //set the id of the debug div
        notificationImage.news.debugPanelID = 'debug-' + notificationImage.imageId;

        //set the id of  the edit button div
        notificationImage.news.editButtonID = 'edit-' + notificationImage.imageId;

        //set the id of  the edit sources div
        notificationImage.news.editSourcesPanelID = 'edit-sources-' + notificationImage.imageId;

        //set the id of  the footers div
        notificationImage.news.footerPanelID = 'footer-' + notificationImage.imageId;

        //set the id of  the footers div
        notificationImage.news.footerPanelID = 'footer-' + notificationImage.imageId;

        //set the id of the options div
        notificationImage.news.optionsPanelID = 'options-' + notificationImage.imageId;

        //set the id of  the edit button div
        notificationImage.news.allReadButtonID = 'read-' + notificationImage.imageId;


        var _containerBox = notificationImage.boxId || "";
        var _title = notificationImage.news.title || _config.Title
        var _content = notificationImage.news.html || "";
        var _footer = notificationImage.news.footer || _config.Footer;

        var _htmlContent;

        _htmlContent =
            '<div class="nd-notification-box-container">' +
                '<div class="nd-notification-box-header">{Title}</div>' +
                '<div class="nd-notification-box-header-edit" id="' + notificationImage.news.editButtonID + '"></div>' +
                '<div class="nd-notification-box-debug" id="' + notificationImage.news.debugPanelID + '"></div>' +
                '<div class="nd-notification-box-content-edit-sources" id="' + notificationImage.news.editSourcesPanelID + '"></div>' +
                '<div class="nd-notification-box-options" id="' + notificationImage.news.optionsPanelID + '"><div class="nd-notification-box-header-read" id="' + notificationImage.news.allReadButtonID + '"></div></div>' +
                '<div class="nd-notification-box-content" id="' + notificationImage.news.contentPanelID + '">{Content}</div>' +
                '<div class="nd-notification-box-footer" id="' + notificationImage.news.footerPanelID + '">{Footer}</div>' +
                '</div>' +
                '<div class="nd-notification-box-pointer"></div>';

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
     Update the notification of unread items
     */
    var _notificationUpdate = function (image) {
        if (image.news != null && image.news != undefined && image.news.data != undefined) {
            //Update the number of unread items
            var unreadItems = $.grep(image.news.data, function (a) {
                return (!a.disabled && a.read == false);
            });
            if (unreadItems.length > 0) {
                $("#" + image.countId).show();
                $("#" + image.countId).html(unreadItems.length);
            }
            else {
                $("#" + image.countId).hide();
            }
        }
    }

    var _eventNotificationCount = function (control) {
        try {
            control.NotificationCountEventCall(control);
        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.Notification.List.updateControls", err);
        }
    }

    /*
     Update the news in the news box
     */
    var _displayDataHtmlPopup = function (notificationImage) {

        try {

            if (notificationImage.user.authenticated) {

                //Generate the news html
                Netdania.News.Monitor.News.getInstance().WriteNews(notificationImage);

                /*
                Update storage
                 */
                Netdania.News.Monitor.Storage.UpdateNews(notificationImage);

                /*
                 Set the content to the corresponding content panel id
                 */
                $('#' + notificationImage.news.contentPanelID).html(notificationImage.news.html);

                /*
                 Display the debug information
                 */
                _debugInfo(notificationImage);


                /*
                 Display the options information
                 */
                _optionsInfo(notificationImage, true);

                //run verification
                _timeOutNotification(notificationImage);

            }
        }
        catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.UpdateNews", err);
        }
    }

    /*
     Set news read after 5 seconds
     */
    var _timeOutNotification = function (notificationImage) {
        setTimeout(
            function () {
                //update the notification count
                _notificationUpdate(notificationImage);

                //update the news read rows
                $("#" + notificationImage.boxId + " .unread").each(function () {
                    $(this).removeClass("unread");
                });

            }, 5000);

    }

    /*
     Display window for editing the news sources
     If the control is configured with the edit button we load the edit
     button to the edit div
     We bind the click event to the edit source function that display the form
     */
    var _editSources = function (notificationImage) {
        try {
            if (notificationImage.user.authenticated && notificationImage.editSources) {
                var _editButton = $('#' + notificationImage.news.editButtonID);
                var _editButtonText = '<span>[Configure]</span>';
                _editButton.html(_editButtonText);
                _editButton.bind("click", {
                    image: notificationImage
                }, function (event) {
                    $(this).hide();
                    var image = event.data.image;
                    var editSourcesPanel = $('#' + image.news.editSourcesPanelID);
                    editSourcesPanel.show();
                    /*
                     Render the div content
                     */
                    var html =
                        '<div id="container">' +
                            '<h4>Notification Sources:</h4>' +
                            '<ul>';


                    /*
                     For every valid source display a check box
                     */
                    $.each(image.source, function (i) {
                        if (image.source[i].isValid) {
                            var checked = 'checked';
                            if (image.source[i].userDisabled) {
                                checked = '';
                            }
                            var checkBoxID = 'cb' + i + image.imageId;
                            html += '<li class="source"><input type="checkbox" id="' + checkBoxID + '" value="' + image.source[i].id + '" ' + checked + ' >' + image.source[i].name + '<span class="ui-icon ui-icon-info source-info" title="' + image.source[i].description + '"></span>';
                            html += '<div class="information"></div></li>';
                        }
                    });

                    html += '</ul>';

                    /*
                     Close button
                     */
                    html += '<div id="button"><input type="button" value="Close" id="ButtonClose"></div>' +
                        '</div>';

                    /*
                     Set the content to the DOM
                     */
                    editSourcesPanel.html(html);


                    /*
                     Set the checked events for all checkboxes
                     */


                    editSourcesPanel.find('.source-info').bind("click", function (event) {
                        var text = $(this).attr('title');
                        var messageDiv = $(this).parent().find('.information');
                        messageDiv.html(text);
                        messageDiv.toggle();
                    });

                    editSourcesPanel.find(':input[type=checkbox]').bind("click", {
                        image: notificationImage
                    }, function (event) {
                        try {
                            //the ID of the news from the checkbox
                            var SourceValue = $(this).attr('value');
                            //checked status of the checkbox
                            var checked = $(this).is(':checked');
                            var idSource = parseInt(SourceValue, 10);
                            event.data.image.source[idSource].userDisabled = checked;

                            $.each(event.data.image.source, function (i) {
                                if (event.data.image.source[i].id == idSource) {
                                    event.data.image.source[i].userDisabled = !checked;

                                    /*
                                     Make the news status disabled/enabled
                                     */
                                  /*  $.each(event.data.image.news.data, function (s) {
                                        if (event.data.image.source[i].source == event.data.image.news.data[s].source && event.data.image.source[i].provider == event.data.image.news.data[s].provider) {
                                            event.data.image.news.data[s].disabled = !checked;
                                        }
                                    });*/

                                    Netdania.News.Monitor.Notification.CheckForValidSources(event.data.image);

                                    /*
                                     Set the disabled status to the local storage
                                     */
                                    Netdania.News.Monitor.Storage.UpdateNews(event.data.image);

                                    /*
                                     Make the news search request to the server
                                     */
                                    if (event.data.image.news.lastNewsTime != null && event.data.image.news.lastNewsTime != undefined) {
                                        /*
                                         If we have already news we get the news from current time to the most recent news available at the client
                                         */
                                        var endDate = new Date().getTime() / 1000;
                                        var startDate = _getSourceLatestNewsTime({control: event.data.image, source: event.data.image.source[i]}) + 1; //event.data.image.news.lastNewsTime + 1;
                                        event.data.image.searchCallback({control: event.data.image, headlines: _config.searchMaxNews, start: startDate, end: endDate});

                                    }

                                    /*Make the news monitor request*/
                                    event.data.image.headlineCallback({control: event.data.image, headlines: event.data.image.maxRequestEvents});

                                    /*Call the news sources changed event*/
                                    event.data.image.NotificationSourcesEventCall(event.data.image);


                                    /*Update displayed news*/
                                    _displayDataHtmlPopup(event.data.image);

                                    return false;
                                }
                            });

                        } catch (err) {
                            Netdania.Util.LogError("Netdania.News.Monitor._editSources source checkbox click event", err);
                        }
                    });

                    /*
                     Set the event for the close button
                     */
                    editSourcesPanel.find('#ButtonClose').bind("click", {
                        image: notificationImage
                    }, function (event) {
                        _editButton.show();
                        editSourcesPanel.hide();
                    });

                });
            }
        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor._editSources", err);
        }
    }

    var _getSourceLatestNewsTime = function (data) {
        var newsTime = 0;
        $.each(data.control.news.data, function (n) {
            if (n.source == data.source && n.t > newsTime) {
                newsTime = n.t;
            }
        });
        return newsTime;
    }

    var _hideNewsBox = function (notificationImage) {
        $("#" + notificationImage.boxId).hide();
        notificationImage.visible = false;
    }

    var _dataUpdatePopup = function (image) {
        try {
            _notificationUpdate(image);
            if (image.visible) {
                _displayDataHtmlPopup(image);
            }
        } catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.DataUpdatePopup", err);
        }
    };

    /*
     Display the news box on user click
     */
    var _displayContainerHtmlPopup = function (notificationImage) {
        try {

            if (!notificationImage.user.authenticated) {
                notificationImage.news.html =  _config.signInLinkDescriptionPopup;
                notificationImage.news.footer = _config.signInLinkFooterPopup;
            }

            //Add news html to the container
            _displayBoxPopup(notificationImage);

            /*
            Display the edit sources information
            */
            _editSources(notificationImage);


            //Display the box container
            $("#" + notificationImage.boxId).show();
            notificationImage.visible = true;

            $(document.body).mouseup(function (event) {
                var target = $(event.target);
                // Clicked outside
                if (!target.parents().andSelf().is('#' + notificationImage.boxId) && !target.parents().andSelf().is('#' + notificationImage.imageId)) {
                    _hideNewsBox(notificationImage);
                }
            });

           /* var _user = new Netdania.Util.User();
            if (_user.isAuthenticated()) {

                //Generate the news html
                Netdania.News.Monitor.News.getInstance().WriteNews(notificationImage);

                //update the news to the storage
                Netdania.News.Monitor.Storage.UpdateNews(notificationImage);

                //Add news html to the container
                _displayBoxPopup(notificationImage);

                *//*
                 Display the debug information
                 *//*
                _debugInfo(notificationImage);

                *//*
                 Display the edit sources information
                 *//*
                _editSources(notificationImage);

            } else {
                *//*
                If the user is not authenticated we display the login message
                 *//*

                notificationImage.news.html = "<div>Please sign in to view notifications</div>";
                notificationImage.news.footer = '<a href="/UI/UserLogin.aspx">Sign in here</a>';

                //Add news html to the container
                _displayBoxPopup(notificationImage);

                *//*
                 Display the debug information
                 *//*
                _debugInfo(notificationImage);
            }


            notificationImage.news.html = "";

            //Display the box container
            $("#" + notificationImage.boxId).show();
            notificationImage.visible = true;

            $(document.body).mouseup(function (event) {
                var target = $(event.target);
                // Clicked outside
                if (!target.parents().andSelf().is('#' + notificationImage.boxId) && !target.parents().andSelf().is('#' + notificationImage.imageId)) {
                    _hideNewsBox(notificationImage);
                }
            });

            //Run the animation timeout
            _timeOutNotification(notificationImage);
*/
        }
        catch (err) {
            Netdania.Util.LogError("Netdania.News.Monitor.DisplayNews", err);
        }
    };

    return {
        Init: function (image) {
            try {
                image.control.addListener(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_UPDATE, function (control) {
                    if (control != null & control != undefined) {
                        _dataUpdatePopup(control);
                    }
                });

                image.control.addListener(Netdania.News.Monitor.Notification.Events.NOTIFICATION_NEWS_COUNT_UPDATE, function (control) {
                    if (control != null & control != undefined) {
                       _dataUpdatePopup(control);
                    }
                });

                _notificationUpdate(image);
            }
            catch (err) {
                Netdania.Util.LogError("Netdania.News.Monitor.Popup.Init", err);
            }
        },
        OnClick: function (image, event) {
            try {
                if (event && event.stopPropagation) {
                    event.stopPropagation();
                }
                else {
                    event.returnValue = false;
                }
                var notificationImage = $.grep(_config.notificationImages, function (a) {
                    return a.imageId == image.id;
                })[0];

                if (!notificationImage.visible) {

                    //write the news html to the box
                    _displayContainerHtmlPopup(notificationImage);

                    _displayDataHtmlPopup(notificationImage);

                }
                else {
                    _hideNewsBox(notificationImage);

                }

            }
            catch (err) {
                Netdania.Util.LogError("Netdania.News.Monitor.OnNotificationPopupClick", err);
            }
        }
    }


})();
















