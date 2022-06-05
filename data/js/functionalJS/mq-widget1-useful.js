
        if(m3 && m3.$('helplink')){
            var messages = m3.util.Message._messages,
                first,
                helpText,
                i;
            
            if(messages[USER.settings.locale]){
                helpText = messages[USER.settings.locale].messages.Help;
            }else{
                for (i in messages) {
                    if (messages.hasOwnProperty(i) && typeof(i) !== 'function') {
                        first = messages[i];
                        break;
                    }
                }
                helpText = first.messages.Help;            
            }
                
            m3.$('helplink').innerHTML = helpText;
        }
    