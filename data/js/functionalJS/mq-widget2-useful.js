    
           //bypass the AOL refresh.
            //m3.isAOL = 0;
    
        // slow loader
        setTimeout(function() {
            var pm  = document.getElementById("pageMask"),
                txt = '<div style="padding: 100px; color: #434343; line-height: 1.5em; font-size: 12px;">' +
                    '<div style="font-size: 16px; font-weight: bold;">MapQuest is loading ...</div>' +
                    '<div style="margin-top: 15px;">If you\'re in a hurry you can jump over to MapQuest Classic and also get maps and directions there:</div>' +
                    '<a href="/_betaoptin/old/">Go to MapQuest Classic Â»</a>' +
                '</div>';
    
            if (pm && pm.style.display == "block") {
                pm.innerHTML = txt;
                if (m3 && m3.util && m3.util.Event) {
                    m3.util.Event.publish('EventLog', {action: 'SlowLoadMessageDisplay'});
                }
            }
        },13000);
    
    