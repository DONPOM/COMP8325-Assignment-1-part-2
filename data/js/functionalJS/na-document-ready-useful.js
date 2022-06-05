
    $(document).ready(function () {

        var _user = new Netdania.Util.User();
        var userCountry = _user.countryName();
        var newsCountry = (userCountry == '' ? 'Domestic' : userCountry);
        Netdania.News.Monitor.Notification.Init(
          { container: 'nd-news-notification-container',
              s: [
                 { source: 'AktiefokusMain_Alerts', provider: 'aktiefokus', name: 'Breaking News on Danish Stocks', description: 'Breaking news in realtime on Danish stocks. Stay updated 24 x 7 and receive same breaking news in NetDania mobile apps for iPhone, iPad and Android phones and tablets.', login: true, useIfLoggedIn: true, countryRestriction: 'dk' },
                 { source: 'fxWire@NC-EM', provider: 'fxwire_dl', name: newsCountry + ' Breaking News D:15', description: 'Breaking news 15 minuted delayed from your country. Stay updated 24 x 7 and receive same breaking news in NetDania mobile apps for iPhone, iPad and Android phones and tablets.', login: false, useIfLoggedIn: false, countryFilter: true },
                 { source: 'fxWire@NC-EM', provider: 'fxwire', name: newsCountry + ' Breaking News', description: 'Breaking news in realtime from your country. Stay updated 24 x 7 and receive same breaking news in NetDania mobile apps for iPhone, iPad and Android phones and tablets.', login: true, useIfLoggedIn: true, countryFilter: true },
                 { source: 'FX_Basic_Alerts', provider: 'mni', name: 'International Breaking News', description: 'Breaking news in realtime from G8 countries. Stay updated 24 x 7 and receive same breaking news in NetDania mobile apps for iPhone, iPad and Android phones and tablets.', login: false,  useIfLoggedIn: true },
                 { source: 'Triggered', provider: 'NetdaniaAlerts', name: 'Netdania Notify', description: 'Set price and trend-line alerts on any instrument and receive the alert when triggered. Stay update 24 x 7 and receive same alerts in NetDania mobile apps for iPhone, iPad and Android phones and tablets.', login: true, useIfLoggedIn: true, alert: true }
                 //{ source: 'Channel1_news_test', provider: 'netdania_newschannel', name: 'Test Channel for Denmark', description: 'Test channel for Denmark users', login: false, useIfLoggedIn: true, countryRestriction: 'dk' }
                ]
          });


        var searchConfig = {

            providers: ['netdania_fxa|FOREX|0|NetDania',
                                     'idc_rtb|BATS|1;2;4;6;7|BATS',
                                        'idc_rtb|DOWJ|1;2;4;6;7|NASDAQ',
                                        'idc_dla|XCSE|1;2;4;6;7|OMX',
                                        'idc_rtb|CHIXU|1;2;4;6;7|CHI-x-U',
                                        'idc_rtb|CHIX|1;2;4;6;7|CHI-x',
                                        'idc_rtb|CHIXC|1;2;4;6;7|CHI-x-C'
                                        ],
            maxResults: 8,
            isMonitor: true,
            containerId: 'lookup',
            callback: function (instrument) {
                document.location.href = "/Products/FullQuote/FullQuote.aspx?symbol=" + instrument.symbol + "&provider=" + instrument.provider + "&ext=" + (instrument.category == 'Stocks' && instrument.country == 'Denmark') + "&d=" + instrument.directory;
                //document.location.href = "/Products/FinanceChart/FinanceChart.aspx?symbol=" + instrument.symbol + "|" + instrument.provider + "&name=" + instrument.label;
                //alert(instrument.isin + "|" + instrument.directory);
            },
            theme: 'white',
            lookupSources: [
                                 {
                                     req_type: 0,
                                     directory: "netdania_forex_lf",
                                     label: "Forex"
                                     //markets: "[ALDP, ALXB, AMTS, APXL]",
                                     //providers: "[idc_dla, idc_rtb]",
                                     //index: 15,
                                     //count: 24
                                 },
                                  {
                                      req_type: 0,
                                      directory: "forex_forwards",
                                      label: "Forex Forwards"
                                      //markets: "[ALDP, ALXB, AMTS, APXL]",
                                      //providers: "[idc_dla, idc_rtb]",
                                      //index: 15,
                                      //count: 24
                                  },
                                {
                                    req_type: 0,
                                    directory: "netdania_stocks",
                                    label: "Stocks"
                                    //markets: "[ALDP, ALXB, AMTS, APXL]",
                                    //providers: "[idc_dla, idc_rtb]",
                                    //index: 15,
                                    //count: 24
                                },
                                 {
                                     req_type: 0,
                                     directory: "netdania_indices_website",
                                     label: "Indices"
                                     //markets: "[ALDP, ALXB, AMTS, APXL]",
                                     //providers: "[idc_dla, idc_rtb]",
                                     //index: 15,
                                     //count: 24
                                 },
                                 {
                                     req_type: 0,
                                     directory: "netdania_futures",
                                     label: "Futures"
                                     //markets: "[ALDP, ALXB, AMTS, APXL]",
                                     //providers: "[idc_dla, idc_rtb]",
                                     //index: 15,
                                     //count: 24
                                 },
                                {
                                    req_type: 0,
                                    directory: "DJ-LF",
                                    label: "DJ Indexes Delayed"
                                }


                            ]
        }
        var lookupComponent = new Netdania.LookupComponent.RenderLookupSearch(searchConfig);
        lookupComponent.initLookupSearch();

    });
