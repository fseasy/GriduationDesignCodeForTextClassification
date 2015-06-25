#coding=utf-8
'''
SOURCE FROM : http://blog.csdn.net/yuan892173701/article/details/8731490
名称　　Unicode　符号

句号　　　 3002　　。　　　

问号　　　 FF1F　　？　　　

叹号　　     FF01　　！　　　


逗号　　　 FF0C　　，

顿号　　　 3001　　、　　　

分号　　　 FF1B　　；　　　

冒号　　　 FF1A　　：　　　


引号　　　 300C　　「　　　

　　　　　 300D　　」　　　

引号　　　 300E　　『　　　

　　　　　 300F　　』　　　

引号　　　 2018　　‘　　　

　　　　　 2019　　’　　　

引号　　　 201C　　“　　　

　　　　　 201D　　”　　　

括号　　　 FF08　　（　　　

　　　　　 FF09　　）　　　

括号　　　 3014　　〔　　　

　　　　　 3015　　〕　　　

括号　　　 3010　　【　　　

　　　　　 3011　　】　　　


破折号　     2014　　—　　　

省略号  　   2026　　…　　　

连接号　     2013　　–　　　

间隔号　     FF0E　　．　　　

书名号  　   300A　 《　　　

　　　　　  300B　　》　　　

书名号  　   3008　 〈　　　

　　　　　  3009　　〉

中文数字（全角数字）
\uff10 ~ \uff19
中文空格
\u3000 似乎也包含 \ue40c
'''
cnpunctuation=u"\u3002\uFF1F\uFF01\uFF0C\u3001\uFF1B\uFF1A\u300C\u300D\u300E\u300F\u2018\u2019\u201C\u201D\uFF08\uFF09\u3014\u3015\u3010\u3011\u2014\u2026\u2013\uFF0E\u300A\u300B\u3008\u3009"
cndigit=u"\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19"
cnblank=u"\u3000\ue40c"
