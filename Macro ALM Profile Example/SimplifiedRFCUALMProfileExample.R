install.packages("ggplot2")

data()
?alm_macro_final

  
ggplot(data = alm_macro_final, 
       mapping = aes(x = Time, y = Value, fill = Class))+
  geom_col(position = "dodge")+
  scale_fill_manual(breaks = c("Assets", "Gap", "Liabilities"),
    values = c("#41905b","#bababa","#d00b0b"))+
  labs(title = "Simplified RFCU ALM Profile",
       y = "Value (Millions USD)",
       x = "Time Buckets")


