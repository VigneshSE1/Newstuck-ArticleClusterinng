# Azure Container Instance Deployment
## Adding Images to  Azure Container Registry
Create images on local system using the `docker-compose.yml` file. Run,

    docker-compose -f docker-compose.yml up --build -d
	
Push the images to ACR

    az login
	az acr login --name prodonemindindiaacr --username <username> --password <password>
	docker-compose push 

## Creating Container Group in ACI

    az container create --resource-group  stage-omi-newstuck-rg --file stage-deploy-aci.yaml

Details of the container group can be found in stage-deploy-aci.yaml

> Note: `az create` uses cached images to run containers. Once a container group is created, any changes made to the ACR images won't be reflected. To update, delete the container group and create again.

To delete the container group, use:

    az container delete --name MyContainerGroup --resource-group MyResourceGroup

 

To check the logs, use:

    az container logs --resource-group  stage-omi-newstuck-rg --name newstuckContainerGroup --container-name stage-omi-clustering-aci



