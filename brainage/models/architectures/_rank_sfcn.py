from brainage.models.architectures import SFCN

class RankSFCNModel(SFCN):
    
    def __init__(self):
        super(SFCN, self)
        
    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        return x

    
    # def forward(
    #     self,
    #     x):

    #     out = list()
    #     x_f = self.feature_extractor(x)
    #     x = self.classifier(x_f)
    #     # x = F.log_softmax(x, dim=1)
    #     out.append(x)