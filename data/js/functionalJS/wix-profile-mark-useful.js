
        if (window.msWriteProfilerMark || lte9)
        {
            new FontLoadDetect(['Helvetica35','Helvetica45','Helvetica55','Helvetica65','Helvetica75'],function ()
            {
                document.getElementsByTagName('html')[0].setAttribute('iefontsloaded','true');
            });
        }
        if (browser_info.indexOf("Chrome") == 0 || browser_info.indexOf("Safari") == 0)
        {
            windoeScroll = { style: document.body };
        } else
        {
            windoeScroll = { style: document.getElementsByTagName('html')[0] };
        }
    