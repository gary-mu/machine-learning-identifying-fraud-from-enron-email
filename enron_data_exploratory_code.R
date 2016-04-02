require(GGally)
enron = read.csv("/Users/garymu/Documents/mini-project/ud120-projects/final_project/enron.csv",
                 stringsAsFactors = F)
names(enron)[1] = 'name'
enron = (subset(enron, enron$name != "TOTAL"))
enron$poi = factor(enron$poi)

include = c()
for (col in (1:length(names(enron)))){
  if(sum(!is.na(enron[col]))/nrow(enron) > 0.5){
    include = append(include, col)
  }
}

msg = c(3,9,14,16,22)
finance = c(2, 4,5,6,7,8,10,11,12,13,18,19,20)
msg_new =c()
finance_new=c()
for (i in msg){
  if (i %in% include){
    msg_new = append(msg_new, i)
  }
}
msg_new

for (i in finance){
  if (i %in% include){
    finance_new = append(finance_new, i)
  }
}
finance_new

ggpairs(enron, columns = finance_new, color = 'poi')
ggpairs(enron, columns = msg_new, color = 'poi' , axisLabels = 'internal')
