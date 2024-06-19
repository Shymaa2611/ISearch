from django.db import models



model=[('Document Term Matrix OR Bitwise','Document Term Matrix OR Bitwise'),
       ('Document Term Matrix AND Bitwise','Document Term Matrix AND Bitwise'),
       ('Inverted Index OR','Inverted Index OR'),
       ('Inverted Index AND','Inverted Index AND'),
       ('TFID','TFID')
       ]



class SearchEngineModel(models.Model):
    query=models.CharField(max_length=500)
    model=models.CharField(max_length=100,choices=model)
  

class documentModel(models.Model):
    document=models.TextField()



